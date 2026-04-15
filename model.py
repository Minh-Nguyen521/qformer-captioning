"""
Full BLIP-2-style model: two training stages.

Stage 1 — Vision-Language Representation Learning
  Loss = ITC + ITM + ITG  |  Trained: Q-Former  |  Frozen: vision encoder

Stage 2 — Vision-to-Language Generative Training
  Loss = LM cross-entropy  |  Trained: FC projection (+ optionally T5)
  Frozen: vision encoder + Q-Former

Inference flow (Stage 2):
  image → CLIPVisionModel → vision_proj → QFormer.get_query_features()
        → qformer_to_lm  → prepend to T5 encoder → caption
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionModel,
    T5ForConditionalGeneration,
    AutoTokenizer,
    CLIPImageProcessor,
)
from qformer import QFormer

# Shared Q-Former defaults — override in build_stage1 / build_stage2 as needed
_QF_DEFAULTS = dict(
    hidden_size        = 768,
    num_attention_heads= 12,
    num_hidden_layers  = 12,
    intermediate_size  = 3072,
    attention_dropout  = 0.1,
    hidden_dropout     = 0.1,
    num_query_tokens   = 32,
    cross_attention_freq = 2,
    max_text_length    = 128,
)


# ── Stage 1 ────────────────────────────────────────────────────────────────────

class Stage1Model(nn.Module):
    """Vision encoder + Q-Former trained with ITC + ITM + ITG."""

    def __init__(
        self,
        vocab_size: int,
        vision_model: str         = "openai/clip-vit-large-patch14",
        freeze_vision: bool       = True,
        # Q-Former architecture
        qf_hidden_size: int       = 768,
        qf_num_heads: int         = 12,
        qf_num_layers: int        = 12,
        qf_intermediate_size: int = 3072,
        qf_attn_dropout: float    = 0.1,
        qf_hidden_dropout: float  = 0.1,
        num_query_tokens: int     = 32,
        cross_attention_freq: int = 2,
        # ITC training
        temperature_init: float   = 0.07,
        queue_size: int           = 65536,
        momentum: float           = 0.995,
        # Loss weights
        itc_weight: float         = 1.0,
        itm_weight: float         = 1.0,
        itg_weight: float         = 1.0,
    ):
        super().__init__()
        self.freeze_vision = freeze_vision
        self.itc_weight = itc_weight
        self.itm_weight = itm_weight
        self.itg_weight = itg_weight
        self.momentum   = momentum
        self.queue_size = queue_size

        # Vision encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model)
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad_(False)

        clip_dim = self.vision_encoder.config.hidden_size
        self.vision_proj = nn.Linear(clip_dim, qf_hidden_size)

        # Q-Former
        self.qformer = QFormer(
            vocab_size,
            hidden_size         = qf_hidden_size,
            num_attention_heads = qf_num_heads,
            num_hidden_layers   = qf_num_layers,
            intermediate_size   = qf_intermediate_size,
            attention_dropout   = qf_attn_dropout,
            hidden_dropout      = qf_hidden_dropout,
            num_query_tokens    = num_query_tokens,
            cross_attention_freq= cross_attention_freq,
        )

        # Momentum encoder for ITC (MoCo-style)
        if queue_size > 0:
            self.vision_encoder_m = CLIPVisionModel.from_pretrained(vision_model)
            self.vision_proj_m    = nn.Linear(clip_dim, qf_hidden_size)
            self.qformer_m        = QFormer(
                vocab_size,
                hidden_size         = qf_hidden_size,
                num_attention_heads = qf_num_heads,
                num_hidden_layers   = qf_num_layers,
                intermediate_size   = qf_intermediate_size,
                attention_dropout   = qf_attn_dropout,
                hidden_dropout      = qf_hidden_dropout,
                num_query_tokens    = num_query_tokens,
                cross_attention_freq= cross_attention_freq,
            )
            self._copy_to_momentum()
            for p in self._momentum_params():
                p.requires_grad_(False)

            self.register_buffer("image_queue",
                F.normalize(torch.randn(256, queue_size), dim=0))
            self.register_buffer("text_queue",
                F.normalize(torch.randn(256, queue_size), dim=0))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.logit_scale = nn.Parameter(torch.ones([]) / temperature_init)

    # ------------------------------------------------------------------
    def _momentum_params(self):
        for m in (self.vision_encoder_m, self.vision_proj_m, self.qformer_m):
            yield from m.parameters()

    def _copy_to_momentum(self):
        for src, tgt in [(self.vision_encoder, self.vision_encoder_m),
                         (self.vision_proj,    self.vision_proj_m),
                         (self.qformer,        self.qformer_m)]:
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.copy_(ps.data)

    @torch.no_grad()
    def _momentum_update(self):
        m = self.momentum
        for src, tgt in [(self.vision_encoder, self.vision_encoder_m),
                         (self.vision_proj,    self.vision_proj_m),
                         (self.qformer,        self.qformer_m)]:
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                pt.data = pt.data * m + ps.data * (1.0 - m)

    @torch.no_grad()
    def _dequeue_enqueue(self, img_feat, txt_feat):
        B   = img_feat.size(0)
        ptr = int(self.queue_ptr)
        if ptr + B > self.queue_size:
            B = self.queue_size - ptr
            img_feat = img_feat[:B]
            txt_feat = txt_feat[:B]
        self.image_queue[:, ptr:ptr+B] = img_feat.T
        self.text_queue[:, ptr:ptr+B]  = txt_feat.T
        self.queue_ptr[0] = (ptr + B) % self.queue_size

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze_vision):
            feats = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        return self.vision_proj(feats)

    @torch.no_grad()
    def _encode_image_m(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_proj_m(
            self.vision_encoder_m(pixel_values=pixel_values).last_hidden_state
        )

    # ------------------------------------------------------------------
    def forward(self, pixel_values, input_ids, attention_mask, labels) -> dict:
        image_features = self._encode_image(pixel_values)

        # ITC
        image_embed = self.qformer.forward_itc_image(image_features)
        text_embed  = self.qformer.forward_itc_text(input_ids, attention_mask)
        sim_i2t = self.logit_scale * image_embed @ text_embed.T
        sim_t2i = sim_i2t.T

        if self.queue_size > 0:
            self._momentum_update()
            image_embed_m = self.qformer_m.forward_itc_image(self._encode_image_m(pixel_values))
            text_embed_m  = self.qformer_m.forward_itc_text(input_ids, attention_mask)
            queue_i = self.image_queue.clone().detach().T
            queue_t = self.text_queue.clone().detach().T
            sim_i2t_all = self.logit_scale * image_embed @ torch.cat([text_embed_m, queue_t], 0).T
            sim_t2i_all = self.logit_scale * text_embed  @ torch.cat([image_embed_m, queue_i], 0).T
            self._dequeue_enqueue(image_embed_m, text_embed_m)
            B       = pixel_values.size(0)
            targets = torch.arange(B, device=pixel_values.device)
            itc_loss = (F.cross_entropy(sim_i2t_all, targets) +
                        F.cross_entropy(sim_t2i_all, targets)) / 2
        else:
            B       = pixel_values.size(0)
            targets = torch.arange(B, device=pixel_values.device)
            itc_loss = (F.cross_entropy(sim_i2t, targets) +
                        F.cross_entropy(sim_t2i, targets)) / 2

        # ITM — hard negative mining
        with torch.no_grad():
            eye = torch.eye(B, device=sim_i2t.device) * 1e4
            w_i2t = F.softmax(sim_i2t - eye, dim=1)
            w_t2i = F.softmax(sim_t2i - eye, dim=1)
        neg_tid = torch.multinomial(w_i2t, 1).squeeze(1)
        neg_iid = torch.multinomial(w_t2i, 1).squeeze(1)

        pos_logits      = self.qformer.forward_itm(image_features, input_ids, attention_mask)
        neg_logits_i2t  = self.qformer.forward_itm(image_features, input_ids[neg_tid], attention_mask[neg_tid])
        neg_logits_t2i  = self.qformer.forward_itm(image_features[neg_iid], input_ids, attention_mask)
        itm_logits  = torch.cat([pos_logits, neg_logits_i2t, neg_logits_t2i], 0)
        itm_targets = torch.cat([torch.ones(B), torch.zeros(B), torch.zeros(B)]).long().to(pixel_values.device)
        itm_loss    = F.cross_entropy(itm_logits, itm_targets)

        # ITG
        itg_loss, _ = self.qformer.forward_itg(image_features, input_ids, attention_mask, labels)

        loss = self.itc_weight * itc_loss + self.itm_weight * itm_loss + self.itg_weight * itg_loss
        return {"loss": loss, "itc_loss": itc_loss.detach(),
                "itm_loss": itm_loss.detach(), "itg_loss": itg_loss.detach()}


# ── Stage 2 ────────────────────────────────────────────────────────────────────

class Stage2Model(nn.Module):
    """Frozen vision encoder + Q-Former + FC projection + Flan-T5."""

    def __init__(
        self,
        vocab_size: int,
        vision_model: str         = "openai/clip-vit-large-patch14",
        lm_model: str             = "google/flan-t5-base",
        freeze_lm: bool           = True,
        # Q-Former architecture
        qf_hidden_size: int       = 768,
        qf_num_heads: int         = 12,
        qf_num_layers: int        = 12,
        qf_intermediate_size: int = 3072,
        qf_attn_dropout: float    = 0.1,
        qf_hidden_dropout: float  = 0.1,
        num_query_tokens: int     = 32,
        cross_attention_freq: int = 2,
        # Generation defaults
        max_new_tokens: int       = 64,
        num_beams: int            = 4,
    ):
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.num_beams      = num_beams

        # Vision encoder (always frozen in Stage 2)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model)
        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)

        clip_dim = self.vision_encoder.config.hidden_size
        self.vision_proj = nn.Linear(clip_dim, qf_hidden_size)

        # Q-Former
        self.qformer = QFormer(
            vocab_size,
            hidden_size         = qf_hidden_size,
            num_attention_heads = qf_num_heads,
            num_hidden_layers   = qf_num_layers,
            intermediate_size   = qf_intermediate_size,
            attention_dropout   = qf_attn_dropout,
            hidden_dropout      = qf_hidden_dropout,
            num_query_tokens    = num_query_tokens,
            cross_attention_freq= cross_attention_freq,
        )

        # Projection: Q-Former hidden → T5 hidden
        lm = T5ForConditionalGeneration.from_pretrained(lm_model)
        self.qformer_to_lm   = nn.Linear(qf_hidden_size, lm.config.d_model)
        self.language_model  = lm
        if freeze_lm:
            for p in self.language_model.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    def _visual_prompt(self, pixel_values: torch.Tensor) -> tuple:
        with torch.no_grad():
            feats = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        image_features = self.vision_proj(feats)
        query_feats    = self.qformer.get_query_features(image_features)
        visual_embeds  = self.qformer_to_lm(query_feats)
        B, Q, _        = visual_embeds.shape
        visual_mask    = torch.ones(B, Q, device=pixel_values.device, dtype=torch.long)
        return visual_embeds, visual_mask

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        visual_embeds, visual_mask = self._visual_prompt(pixel_values)
        text_embeds   = self.language_model.encoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        full_mask     = torch.cat([visual_mask, attention_mask], dim=1)
        return self.language_model(
            inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels
        )

    @torch.no_grad()
    def generate(self, pixel_values, tokenizer, prompt="describe the image:", **gen_kwargs):
        B = pixel_values.size(0)
        visual_embeds, visual_mask = self._visual_prompt(pixel_values)
        prompt_enc    = tokenizer([prompt] * B, return_tensors="pt",
                                  add_special_tokens=True).to(pixel_values.device)
        text_embeds   = self.language_model.encoder.embed_tokens(prompt_enc.input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        full_mask     = torch.cat([visual_mask, prompt_enc.attention_mask], dim=1)
        ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens      = gen_kwargs.get("max_new_tokens", self.max_new_tokens),
            num_beams           = gen_kwargs.get("num_beams",       self.num_beams),
            length_penalty      = gen_kwargs.get("length_penalty",  1.0),
            no_repeat_ngram_size= gen_kwargs.get("no_repeat_ngram_size", 3),
            early_stopping=True,
        )
        return tokenizer.batch_decode(ids, skip_special_tokens=True)

    def load_stage1_weights(self, checkpoint_path: str, device):
        ckpt  = torch.load(checkpoint_path, map_location=device)
        state = ckpt["model_state_dict"]
        keys  = {k: v for k, v in state.items()
                 if k.startswith(("vision_encoder.", "vision_proj.", "qformer."))}
        missing, unexpected = self.load_state_dict(keys, strict=False)
        print(f"  Stage 1 weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")


# ── Factories ──────────────────────────────────────────────────────────────────

def build_stage1(
    vision_model: str         = "openai/clip-vit-large-patch14",
    lm_model: str             = "google/flan-t5-base",
    freeze_vision: bool       = True,
    qf_hidden_size: int       = 768,
    qf_num_heads: int         = 12,
    qf_num_layers: int        = 12,
    qf_intermediate_size: int = 3072,
    qf_attn_dropout: float    = 0.1,
    qf_hidden_dropout: float  = 0.1,
    num_query_tokens: int     = 32,
    cross_attention_freq: int = 2,
    temperature_init: float   = 0.07,
    queue_size: int           = 65536,
    momentum: float           = 0.995,
    itc_weight: float         = 1.0,
    itm_weight: float         = 1.0,
    itg_weight: float         = 1.0,
):
    tokenizer       = AutoTokenizer.from_pretrained(lm_model)
    image_processor = CLIPImageProcessor.from_pretrained(vision_model)
    model = Stage1Model(
        vocab_size          = tokenizer.vocab_size,
        vision_model        = vision_model,
        freeze_vision       = freeze_vision,
        qf_hidden_size      = qf_hidden_size,
        qf_num_heads        = qf_num_heads,
        qf_num_layers       = qf_num_layers,
        qf_intermediate_size= qf_intermediate_size,
        qf_attn_dropout     = qf_attn_dropout,
        qf_hidden_dropout   = qf_hidden_dropout,
        num_query_tokens    = num_query_tokens,
        cross_attention_freq= cross_attention_freq,
        temperature_init    = temperature_init,
        queue_size          = queue_size,
        momentum            = momentum,
        itc_weight          = itc_weight,
        itm_weight          = itm_weight,
        itg_weight          = itg_weight,
    )
    return model, image_processor, tokenizer


def build_stage2(
    vision_model: str         = "openai/clip-vit-large-patch14",
    lm_model: str             = "google/flan-t5-base",
    freeze_lm: bool           = True,
    qf_hidden_size: int       = 768,
    qf_num_heads: int         = 12,
    qf_num_layers: int        = 12,
    qf_intermediate_size: int = 3072,
    qf_attn_dropout: float    = 0.1,
    qf_hidden_dropout: float  = 0.1,
    num_query_tokens: int     = 32,
    cross_attention_freq: int = 2,
    max_new_tokens: int       = 64,
    num_beams: int            = 4,
):
    tokenizer       = AutoTokenizer.from_pretrained(lm_model)
    image_processor = CLIPImageProcessor.from_pretrained(vision_model)
    model = Stage2Model(
        vocab_size          = tokenizer.vocab_size,
        vision_model        = vision_model,
        lm_model            = lm_model,
        freeze_lm           = freeze_lm,
        qf_hidden_size      = qf_hidden_size,
        qf_num_heads        = qf_num_heads,
        qf_num_layers       = qf_num_layers,
        qf_intermediate_size= qf_intermediate_size,
        qf_attn_dropout     = qf_attn_dropout,
        qf_hidden_dropout   = qf_hidden_dropout,
        num_query_tokens    = num_query_tokens,
        cross_attention_freq= cross_attention_freq,
        max_new_tokens      = max_new_tokens,
        num_beams           = num_beams,
    )
    return model, image_processor, tokenizer
