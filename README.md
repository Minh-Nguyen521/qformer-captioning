# Q-Former Image Captioning

BLIP-2-style image captioning using a **Querying Transformer (Q-Former)** to bridge a frozen CLIP vision encoder and a frozen Flan-T5 language model.

## Why two stages?

**Stage 1** teaches the Q-Former *what* visual information is language-relevant, using three losses:
- **ITC** (Image-Text Contrastive) — global alignment between image and text embeddings
- **ITM** (Image-Text Matching) — fine-grained match / no-match classification with hard negatives
- **ITG** (Image-Grounded Text Generation) — causal LM loss, queries as prefix

**Stage 2** teaches *how* to generate fluent text from already-aligned query features. Because the queries are pre-aligned from Stage 1, the frozen T5 converges quickly.

Skipping Stage 1 and training end-to-end from scratch with a frozen LLM leads to degenerate queries that encode shortcuts rather than semantic content.

## Installation

```bash
pip install torch transformers datasets Pillow
# for evaluation metrics (optional)
pip install nltk rouge-score
```

## Quick start

### 1 — Train Stage 1

```bash
python train_stage1.py
```

Default dataset: [`jxie/flickr8k`](https://huggingface.co/datasets/jxie/flickr8k) — 8,000 images, 5 captions each, splits: train / validation / test.

Key options:

```bash
python train_stage1.py \
  --vision_model     openai/clip-vit-large-patch14 \
  --language_model   google/flan-t5-base \
  --epochs           20 \
  --batch_size       32 \
  --lr               1e-4 \
  --num_query_tokens 32 \
  --output_dir       ./checkpoints/qformer_stage1
```

Use a smaller vision encoder for faster iteration:

```bash
python train_stage1.py --vision_model openai/clip-vit-base-patch32 --batch_size 64
```

Smoke test with a small subset (edit `config.py`):

```python
# config.py → DataConfig
hf_max_train_samples = 500
hf_max_val_samples   = 100
```

```bash
python train_stage1.py --epochs 2
```

Checkpoints are saved to `./checkpoints/qformer_stage1/`:
- `latest.pt` — most recent epoch (for resuming)
- `epoch_N.pt` — per-epoch snapshots
- `best.pt` — lowest validation loss
- `final.pt` — end of training

### 2 — Train Stage 2

```bash
python train_stage2.py \
  --stage1_checkpoint ./checkpoints/qformer_stage1/best.pt
```

Key options:

```bash
python train_stage2.py \
  --stage1_checkpoint ./checkpoints/qformer_stage1/best.pt \
  --epochs     10 \
  --batch_size 16 \
  --lr         1e-5 \
  --output_dir ./checkpoints/qformer_stage2

# Unfreeze the language model for higher quality (costs more memory)
python train_stage2.py \
  --stage1_checkpoint ./checkpoints/qformer_stage1/best.pt \
  --unfreeze_lm --lr 5e-5
```

### 3 — Inference

```bash
# Single image
python inference.py \
  --checkpoint ./checkpoints/qformer_stage2/best.pt \
  --images photo.jpg

# Multiple images
python inference.py \
  --checkpoint ./checkpoints/qformer_stage2/best.pt \
  --images img1.jpg img2.jpg img3.png

# Whole directory
python inference.py \
  --checkpoint ./checkpoints/qformer_stage2/best.pt \
  --image_dir  ./my_photos/

# Custom generation settings
python inference.py \
  --checkpoint     ./checkpoints/qformer_stage2/best.pt \
  --images         photo.jpg \
  --num_beams      8 \
  --max_new_tokens 80 \
  --prompt         "a photo of"
```

---

## Datasets

### Flickr8k (default)

[`jxie/flickr8k`](https://huggingface.co/datasets/jxie/flickr8k) — 8k images, 5 captions per image stored in separate columns (`caption_0` … `caption_4`).

| Split | Images | Samples (×5 captions) |
|---|---|---|
| train | 6,000 | 30,000 |
| validation | 1,000 | 5,000 |
| test | 1,000 | 5,000 |

This is the default — no config changes needed:

```bash
python train_stage1.py   # uses jxie/flickr8k out of the box
```

Equivalent explicit config:

```python
DataConfig(
    hf_dataset      = "jxie/flickr8k",
    hf_caption_cols = ["caption_0", "caption_1", "caption_2", "caption_3", "caption_4"],
    hf_image_col    = "image",
)
```

### Other HuggingFace datasets

**COCO Captions** (captions stored as list of dicts):
```python
DataConfig(
    hf_dataset      = "HuggingFaceM4/COCO",
    hf_config       = "2017_captions",
    hf_caption_cols = None,              # disable multi-column mode
    hf_caption_col  = "sentences_raw",
    hf_caption_key  = "raw",            # dict key: {"raw": "...", "tokens": [...]}
)
```

**Flickr30k** (single string caption column):
```python
DataConfig(
    hf_dataset      = "nlphuji/flickr30k",
    hf_caption_cols = None,
    hf_caption_col  = "caption",
    hf_caption_key  = None,
)
```

### Streaming mode (large datasets, low memory)

```python
DataConfig(hf_streaming=True)
```

Shuffle is handled by the dataset with a buffer — the DataLoader does not shuffle.

### Custom CSV

Bypass HuggingFace with a local CSV file:

```
image_path,caption
/data/images/dog.jpg,a golden retriever playing fetch
/data/images/city.jpg,busy street in downtown manhattan
```

```python
DataConfig(custom_csv="./my_data/captions.csv", train_split=0.9)
```

`image_path` can be absolute or relative to the CSV file's directory.

---

## Model variants

| Vision encoder | LM | GPU memory | Notes |
|---|---|---|---|
| `openai/clip-vit-base-patch32` | `google/flan-t5-base` | ~8 GB | fast baseline |
| `openai/clip-vit-large-patch14` | `google/flan-t5-base` | ~14 GB | recommended |
| `openai/clip-vit-large-patch14` | `google/flan-t5-large` | ~20 GB | higher quality |
| `openai/clip-vit-large-patch14` | `google/flan-t5-xl` | ~36 GB | best |

Change via CLI flags or `config.py`.

---

## Resuming training

Both train scripts automatically resume from `latest.pt` if it exists:

```bash
# Just re-run — picks up where it left off
python train_stage1.py --output_dir ./checkpoints/qformer_stage1
```

---

## Config reference

```python
Config(
    vision=VisionConfig(
        model_name="openai/clip-vit-large-patch14",
        freeze=True,
    ),
    qformer=QFormerConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_query_tokens=32,       # more queries = more capacity
        cross_attention_freq=2,    # cross-attn every N layers
    ),
    lm=LMConfig(
        model_name="google/flan-t5-base",
        freeze=True,               # set False to fine-tune the LLM
    ),
    data=DataConfig(
        hf_dataset="jxie/flickr8k",
        hf_caption_cols=["caption_0","caption_1","caption_2","caption_3","caption_4"],
        hf_max_train_samples=None, # None = full dataset
    ),
    stage1=Stage1Config(
        num_epochs=20,
        batch_size=32,
        learning_rate=1e-4,
        queue_size=65536,          # MoCo queue, set 0 to disable
    ),
    stage2=Stage2Config(
        num_epochs=10,
        batch_size=16,
        learning_rate=1e-5,
        num_beams=4,
        max_new_tokens=64,
    ),
)
```
# qformer-captioning
