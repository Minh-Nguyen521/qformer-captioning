"""
Dataset classes for Q-Former image captioning.

Primary source: HuggingFace Hub via `datasets.load_dataset()`.
Fallback:       custom CSV with columns [image_path, caption].

Two caption column layouts are supported:

  Multi-column  (hf_caption_cols set)
    e.g. Flickr8k: caption_0, caption_1, ..., caption_4
    Each row expands into N samples, one per column.

  Single-column  (hf_caption_cols=None)
    hf_caption_col value can be:
      - str          → "a dog playing fetch"
      - list[str]    → ["a dog playing", "dog with ball", ...]
      - list[dict]   → [{"raw": "a dog playing", "tokens": [...]}, ...]

Stage 1 item:  pixel_values, input_ids, attention_mask, labels
Stage 2 item:  pixel_values, input_ids (prefix), attention_mask, labels (caption)
"""

import csv
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from PIL import Image
from transformers import CLIPImageProcessor, PreTrainedTokenizer


# ── Tokenization helpers (shared by all dataset classes) ──────────────────────

def _stage1_item(pixel_values, caption, tokenizer, max_length):
    enc = tokenizer(
        caption,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.squeeze(0)
    attention_mask = enc.attention_mask.squeeze(0)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _stage2_item(pixel_values, caption, tokenizer, max_length,
                 task_prefix="describe the image: "):
    enc_input = tokenizer(
        task_prefix,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    label_enc = tokenizer(
        caption,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = label_enc.input_ids.squeeze(0)
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "pixel_values": pixel_values,
        "input_ids": enc_input.input_ids.squeeze(0),
        "attention_mask": enc_input.attention_mask.squeeze(0),
        "labels": labels,
    }


def _pil(image) -> Image.Image:
    """Ensure we always have an RGB PIL Image regardless of HF format."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    import numpy as np
    return Image.fromarray(np.array(image)).convert("RGB")


# ── Caption extraction ─────────────────────────────────────────────────────────

def _extract_caption(value, caption_key: Optional[str], idx: int) -> str:
    """
    Pull a single caption string from the raw dataset value.

    value can be:
      - str
      - list[str]
      - list[dict]  (caption_key specifies which dict key holds the text)
    idx selects which caption when multiple are present.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        item = value[idx % len(value)]
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ([caption_key] if caption_key else []) + ["raw", "caption", "text", "sentence"]:
                if key and key in item:
                    return item[key]
            raise ValueError(f"Cannot find caption text in dict: {item}")
    raise TypeError(f"Unsupported caption format: {type(value)}")


def _count_captions(value) -> int:
    """How many captions does one row have?"""
    if isinstance(value, str):
        return 1
    if isinstance(value, list):
        return len(value)
    return 1


# ── HuggingFace map-style dataset ─────────────────────────────────────────────

class HFCaptionDataset(Dataset):
    """
    Wraps a HuggingFace dataset split (already loaded, not streaming).

    Each row is expanded into N samples — one per caption — so every
    caption is seen during training (5 samples per image for Flickr8k/COCO).

    Caption layout is controlled by two mutually exclusive options:
      caption_cols  — list of column names, one caption each (e.g. Flickr8k)
      caption_col   — single column containing str/list[str]/list[dict]
    """

    def __init__(
        self,
        hf_split,
        image_col: str,
        image_processor: CLIPImageProcessor,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        stage: int = 1,
        task_prefix: str = "describe the image: ",
        max_samples: Optional[int] = None,
        # Multi-column layout (e.g. Flickr8k)
        caption_cols: Optional[list] = None,
        # Single-column layout
        caption_col: str = "caption",
        caption_key: Optional[str] = None,
    ):
        self.hf_split = hf_split
        self.image_col = image_col
        self.caption_cols = caption_cols        # takes priority when set
        self.caption_col = caption_col
        self.caption_key = caption_key
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage = stage
        self.task_prefix = task_prefix

        self.index = self._build_index(max_samples)

    def _build_index(self, max_samples: Optional[int]) -> list[tuple[int, int]]:
        n = len(self.hf_split)
        if n == 0:
            return []

        if self.caption_cols:
            # One sample per caption column — count is fixed and known
            caps_per_row = len(self.caption_cols)
        else:
            # Peek at row 0 to determine per-row caption count
            caps_per_row = _count_captions(self.hf_split[0][self.caption_col])

        index = [(i, j) for i in range(n) for j in range(caps_per_row)]
        if max_samples is not None:
            index = index[:max_samples]
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        row_idx, cap_idx = self.index[idx]
        row = self.hf_split[row_idx]

        image = _pil(row[self.image_col])
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        if self.caption_cols:
            caption = row[self.caption_cols[cap_idx]]   # direct column access
        else:
            caption = _extract_caption(row[self.caption_col], self.caption_key, cap_idx)

        if self.stage == 1:
            return _stage1_item(pixel_values, caption, self.tokenizer, self.max_length)
        else:
            return _stage2_item(pixel_values, caption, self.tokenizer,
                                self.max_length, self.task_prefix)


# ── HuggingFace streaming / IterableDataset ────────────────────────────────────

class HFCaptionIterableDataset(IterableDataset):
    """
    Streaming wrapper — use when hf_streaming=True.
    DataLoader shuffle must be False (shuffle happens server-side via buffer).
    Does NOT support __len__, so progress bars show step count only.
    """

    def __init__(
        self,
        hf_split,
        image_col: str,
        image_processor: CLIPImageProcessor,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        stage: int = 1,
        task_prefix: str = "describe the image: ",
        shuffle_buffer: int = 1000,
        max_samples: Optional[int] = None,
        caption_cols: Optional[list] = None,
        caption_col: str = "caption",
        caption_key: Optional[str] = None,
    ):
        self.hf_split = hf_split.shuffle(buffer_size=shuffle_buffer)
        self.image_col = image_col
        self.caption_cols = caption_cols
        self.caption_col = caption_col
        self.caption_key = caption_key
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage = stage
        self.task_prefix = task_prefix
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        for row in self.hf_split:
            if self.max_samples is not None and count >= self.max_samples:
                break

            image = _pil(row[self.image_col])
            pixel_values = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values.squeeze(0)

            if self.caption_cols:
                captions = [row[col] for col in self.caption_cols]
            else:
                raw = row[self.caption_col]
                captions = [
                    _extract_caption(raw, self.caption_key, i)
                    for i in range(_count_captions(raw))
                ]

            for caption in captions:
                if self.stage == 1:
                    yield _stage1_item(pixel_values, caption, self.tokenizer, self.max_length)
                else:
                    yield _stage2_item(pixel_values, caption, self.tokenizer,
                                       self.max_length, self.task_prefix)
                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    return


# ── Custom CSV fallback ────────────────────────────────────────────────────────

class CSVCaptionDataset(Dataset):
    """
    Fallback when custom_csv is set.  CSV must have header: image_path,caption
    image_path may be absolute or relative to the CSV file's directory.
    """

    def __init__(
        self,
        csv_path: str,
        image_processor: CLIPImageProcessor,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        stage: int = 1,
        task_prefix: str = "describe the image: ",
        indices: Optional[list[int]] = None,
    ):
        csv_dir = Path(csv_path).parent
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage = stage
        self.task_prefix = task_prefix

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        def resolve(p):
            return p if Path(p).is_absolute() else str(csv_dir / p)

        self.samples = [(resolve(r["image_path"]), r["caption"]) for r in rows]
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, caption = self.samples[idx]
        image = _pil(Image.open(image_path))
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)
        if self.stage == 1:
            return _stage1_item(pixel_values, caption, self.tokenizer, self.max_length)
        else:
            return _stage2_item(pixel_values, caption, self.tokenizer,
                                self.max_length, self.task_prefix)


# ── DataLoader builder ─────────────────────────────────────────────────────────

_FLICKR8K_CAPTION_COLS = ["caption_0", "caption_1", "caption_2", "caption_3", "caption_4"]
_PIN_MEMORY = torch.cuda.is_available()


def build_dataloaders(
    image_processor: CLIPImageProcessor,
    tokenizer: PreTrainedTokenizer,
    stage: int,
    train_batch_size: int       = 32,
    eval_batch_size: int        = 64,
    num_workers: int            = 4,
    seed: int                   = 42,
    # HuggingFace dataset
    hf_dataset: str             = "jxie/flickr8k",
    hf_config: Optional[str]    = None,
    hf_train_split: str         = "train",
    hf_val_split: str           = "validation",
    hf_caption_cols: Optional[list] = _FLICKR8K_CAPTION_COLS,
    hf_image_col: str           = "image",
    hf_caption_col: str         = "caption_0",
    hf_caption_key: Optional[str] = None,
    hf_max_train_samples: Optional[int] = None,
    hf_max_val_samples: Optional[int]   = None,
    hf_cache_dir: Optional[str] = None,
    hf_streaming: bool          = False,
    max_caption_length: int     = 128,
    # CSV fallback (set to bypass HF entirely)
    custom_csv: Optional[str]   = None,
    train_split: float          = 0.9,
):
    """
    Build (train_loader, val_loader).

    Priority:
      1. custom_csv  → CSVCaptionDataset
      2. hf_dataset  → HFCaptionDataset (map) or HFCaptionIterableDataset (streaming)
    """

    # ── CSV fallback ───────────────────────────────────────────────────────────
    if custom_csv:
        rng   = random.Random(seed)
        total = len(CSVCaptionDataset(custom_csv, image_processor, tokenizer))
        idxs  = list(range(total))
        rng.shuffle(idxs)
        cut   = int(total * train_split)
        train_ds = CSVCaptionDataset(custom_csv, image_processor, tokenizer,
                                     max_length=max_caption_length, stage=stage,
                                     indices=idxs[:cut])
        val_ds   = CSVCaptionDataset(custom_csv, image_processor, tokenizer,
                                     max_length=max_caption_length, stage=stage,
                                     indices=idxs[cut:])
        train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=_PIN_MEMORY)
        val_loader   = DataLoader(val_ds,   batch_size=eval_batch_size,  shuffle=False,
                                  num_workers=num_workers, pin_memory=_PIN_MEMORY)
        return train_loader, val_loader

    # ── HuggingFace dataset ────────────────────────────────────────────────────
    from datasets import load_dataset

    print(f"Loading '{hf_dataset}'"
          + (f" (config={hf_config})" if hf_config else "")
          + (" [streaming]" if hf_streaming else "") + " ...")

    hf_ds = load_dataset(hf_dataset, hf_config, cache_dir=hf_cache_dir, streaming=hf_streaming)

    common_kwargs = dict(
        image_col     = hf_image_col,
        caption_cols  = hf_caption_cols,
        caption_col   = hf_caption_col,
        caption_key   = hf_caption_key,
        image_processor = image_processor,
        tokenizer     = tokenizer,
        max_length    = max_caption_length,
        stage         = stage,
    )

    if hf_streaming:
        train_ds = HFCaptionIterableDataset(hf_ds[hf_train_split],
                                            max_samples=hf_max_train_samples, **common_kwargs)
        val_ds   = HFCaptionIterableDataset(hf_ds[hf_val_split], shuffle_buffer=100,
                                            max_samples=hf_max_val_samples, **common_kwargs)
        train_loader = DataLoader(train_ds, batch_size=train_batch_size,
                                  num_workers=num_workers, pin_memory=_PIN_MEMORY)
        val_loader   = DataLoader(val_ds,   batch_size=eval_batch_size,
                                  num_workers=num_workers, pin_memory=_PIN_MEMORY)
    else:
        train_ds = HFCaptionDataset(hf_ds[hf_train_split],
                                    max_samples=hf_max_train_samples, **common_kwargs)
        val_ds   = HFCaptionDataset(hf_ds[hf_val_split],
                                    max_samples=hf_max_val_samples, **common_kwargs)
        train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=_PIN_MEMORY)
        val_loader   = DataLoader(val_ds,   batch_size=eval_batch_size,  shuffle=False,
                                  num_workers=num_workers, pin_memory=_PIN_MEMORY)

    print(f"  Train samples: {len(train_ds) if not hf_streaming else '(streaming)'}")
    print(f"  Val   samples: {len(val_ds)   if not hf_streaming else '(streaming)'}")
    return train_loader, val_loader
