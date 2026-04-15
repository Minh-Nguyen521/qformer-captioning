"""
Inference with a trained Stage 2 Q-Former captioning model.

Usage:
    python inference.py --checkpoint ./checkpoints/qformer_stage2/best.pt \
                        --images photo.jpg
    python inference.py --checkpoint ./checkpoints/qformer_stage2/best.pt \
                        --image_dir ./photos/
    python inference.py --checkpoint ./checkpoints/qformer_stage2/best.pt \
                        --images a.jpg b.jpg --num_beams 8 --prompt "a photo of"
"""

import argparse
import glob
from pathlib import Path

import torch
from PIL import Image
from torch.amp import autocast

from model import build_stage2

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def collect_paths(images: list[str], image_dir: str = None) -> list[str]:
    paths = list(images) if images else []
    if image_dir:
        for ext in SUPPORTED_EXTS:
            paths += glob.glob(str(Path(image_dir) / f"*{ext}"))
            paths += glob.glob(str(Path(image_dir) / f"*{ext.upper()}"))
    if not paths:
        raise ValueError("No images found.")
    return sorted(set(paths))


@torch.no_grad()
def caption_images(
    image_paths: list[str],
    checkpoint: str = None,
    vision_model: str = "openai/clip-vit-large-patch14",
    language_model: str = "google/flan-t5-base",
    num_query_tokens: int = 32,
    batch_size: int = 8,
    num_beams: int = 4,
    max_new_tokens: int = 64,
    prompt: str = "describe the image:",
    fp16: bool = True,
) -> list[tuple[str, str]]:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    fp16 = fp16 and device.type == "cuda"

    model, image_processor, tokenizer = build_stage2(
        vision_model     = vision_model,
        lm_model         = language_model,
        num_query_tokens = num_query_tokens,
    )

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("[info] No checkpoint provided — random projection weights.")

    model = model.to(device)
    model.eval()

    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        pixel_values = image_processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device)

        with autocast("cuda", enabled=fp16):
            captions = model.generate(
                pixel_values, tokenizer,
                prompt=prompt,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )

        for path, cap in zip(batch_paths, captions):
            results.append((path, cap))

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--images", nargs="*", default=[])
    p.add_argument("--image_dir", default=None)
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--language_model", default="google/flan-t5-base")
    p.add_argument("--num_query_tokens", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--prompt", default="describe the image:")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    args = p.parse_args()

    paths = collect_paths(args.images, args.image_dir)
    print(f"Captioning {len(paths)} image(s)...\n")

    results = caption_images(
        paths,
        checkpoint=args.checkpoint,
        vision_model=args.vision_model,
        language_model=args.language_model,
        num_query_tokens=args.num_query_tokens,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        fp16=args.fp16,
    )

    print("─" * 60)
    for path, cap in results:
        print(f"Image:   {Path(path).name}")
        print(f"Caption: {cap}")
        print()
