"""
Q-Former: Querying Transformer (BLIP-2 style)

Architecture per layer:
  ┌─────────────────────────────────────────────────────────┐
  │  Self-Attention (queries + text tokens, shared weights) │
  │  ↓ (every `cross_attention_freq` layers only)           │
  │  Cross-Attention (queries attend to image patch feats)  │
  │  ↓                                                      │
  │  Feed-Forward Network                                   │
  └─────────────────────────────────────────────────────────┘

Three forward modes:
  ITC  — queries only,           no text
  ITM  — queries + text,         bidirectional self-attention
  ITG  — queries + text,         causal mask on text positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QFormerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        has_cross_attention: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_attention_heads, dropout=attention_dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)

        self.has_cross_attention = has_cross_attention
        if has_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                hidden_size, num_attention_heads, dropout=attention_dropout, batch_first=True
            )
            self.norm_cross = nn.LayerNorm(hidden_size)

        self.ffn   = FeedForward(hidden_size, intermediate_size, hidden_dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_features: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        num_query_tokens: int = 0,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        x = self.norm1(hidden_states)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        hidden_states = residual + x

        # Cross-attention (queries only → image patches)
        if self.has_cross_attention and image_features is not None:
            queries    = hidden_states[:, :num_query_tokens]
            residual_q = queries
            queries, _ = self.cross_attn(self.norm_cross(queries), image_features, image_features)
            queries    = residual_q + queries
            hidden_states = torch.cat([queries, hidden_states[:, num_query_tokens:]], dim=1)

        # FFN
        residual = hidden_states
        hidden_states = self.ffn(self.norm2(hidden_states)) + residual
        return hidden_states


class QFormer(nn.Module):
    """
    Full Q-Former stack.

    Public methods:
        forward_itc_image(image_features)            → normalized image embed
        forward_itc_text(input_ids, attention_mask)  → normalized text embed
        forward_itm(image_features, input_ids, ...)  → (B, 2) match logits
        forward_itg(image_features, input_ids, ...)  → (loss, logits)
        get_query_features(image_features)            → (B, Q, H)  for Stage 2
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        intermediate_size: int = 3072,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        num_query_tokens: int = 32,
        cross_attention_freq: int = 2,
        max_text_length: int = 128,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        # Text embeddings
        self.text_embeddings     = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_text_length, hidden_size)
        self.text_norm           = nn.LayerNorm(hidden_size)
        self.text_dropout        = nn.Dropout(hidden_dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            QFormerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                has_cross_attention=(i % cross_attention_freq == 0),
            )
            for i in range(num_hidden_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

        # ITC projection heads → shared 256-d space
        self.itc_image_proj = nn.Linear(hidden_size, 256, bias=False)
        self.itc_text_proj  = nn.Linear(hidden_size, 256, bias=False)

        # ITM head
        self.itm_head = nn.Linear(hidden_size, 2)

        # ITG LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        S = input_ids.size(1)
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        emb = self.text_embeddings(input_ids) + self.position_embeddings(positions)
        return self.text_dropout(self.text_norm(emb))

    def _expand_queries(self, B: int, device) -> torch.Tensor:
        return self.query_tokens.expand(B, -1, -1)

    def _run_layers(self, hidden_states, image_features=None,
                    attn_mask=None, key_padding_mask=None, num_query_tokens=0):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                image_features=image_features,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                num_query_tokens=num_query_tokens,
            )
        return self.final_norm(hidden_states)

    def _key_padding_mask(self, B: int, Q: int, attention_mask: torch.Tensor) -> torch.Tensor:
        query_mask = torch.zeros(B, Q, dtype=torch.bool, device=attention_mask.device)
        text_mask  = ~attention_mask.bool()
        return torch.cat([query_mask, text_mask], dim=1)

    def _causal_attn_mask(self, Q: int, S: int, device) -> torch.Tensor:
        total = Q + S
        mask  = torch.zeros(total, total, device=device)
        mask[Q:, Q:] = nn.Transformer.generate_square_subsequent_mask(S, device=device)
        return mask

    # ------------------------------------------------------------------
    # ITC
    # ------------------------------------------------------------------

    def forward_itc_image(self, image_features: torch.Tensor) -> torch.Tensor:
        B = image_features.size(0)
        Q = self.num_query_tokens
        queries   = self._expand_queries(B, image_features.device)
        out       = self._run_layers(queries, image_features=image_features, num_query_tokens=Q)
        img_embed = out[:, :Q].mean(dim=1)
        return F.normalize(self.itc_image_proj(img_embed), dim=-1)

    def forward_itc_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text  = self._embed_text(input_ids)
        kpm   = ~attention_mask.bool()
        out   = self._run_layers(text, key_padding_mask=kpm)
        return F.normalize(self.itc_text_proj(out[:, 0]), dim=-1)

    # ------------------------------------------------------------------
    # ITM
    # ------------------------------------------------------------------

    def forward_itm(self, image_features, input_ids, attention_mask) -> torch.Tensor:
        B = image_features.size(0)
        Q = self.num_query_tokens
        hidden = torch.cat([self._expand_queries(B, image_features.device),
                            self._embed_text(input_ids)], dim=1)
        kpm = self._key_padding_mask(B, Q, attention_mask)
        out = self._run_layers(hidden, image_features=image_features,
                               key_padding_mask=kpm, num_query_tokens=Q)
        return self.itm_head(out[:, 0])

    # ------------------------------------------------------------------
    # ITG
    # ------------------------------------------------------------------

    def forward_itg(self, image_features, input_ids, attention_mask, labels) -> tuple:
        B = image_features.size(0)
        Q = self.num_query_tokens
        S = input_ids.size(1)
        hidden    = torch.cat([self._expand_queries(B, image_features.device),
                               self._embed_text(input_ids)], dim=1)
        attn_mask = self._causal_attn_mask(Q, S, image_features.device)
        kpm       = self._key_padding_mask(B, Q, attention_mask)
        out    = self._run_layers(hidden, image_features=image_features,
                                  attn_mask=attn_mask, key_padding_mask=kpm,
                                  num_query_tokens=Q)
        logits = self.lm_head(out[:, Q:])
        loss   = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return loss, logits

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def get_query_features(self, image_features: torch.Tensor) -> torch.Tensor:
        B = image_features.size(0)
        Q = self.num_query_tokens
        queries = self._expand_queries(B, image_features.device)
        out     = self._run_layers(queries, image_features=image_features, num_query_tokens=Q)
        return out[:, :Q]
