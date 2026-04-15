"""
Stage 1: Vision-Language Representation Learning (ITC + ITM + ITG)

Usage:
    python train_stage1.py
    python train_stage1.py --vision_model openai/clip-vit-base-patch32 --batch_size 64
"""

import argparse
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import set_seed, get_cosine_schedule_with_warmup

from model import build_stage1
from dataset import build_dataloaders


def save_checkpoint(model, optimizer, scaler, epoch, step, path: Path, is_best=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "step": step,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict() if scaler else None,
    }, path)
    if is_best:
        import shutil; shutil.copy(path, path.parent / "best.pt")
    print(f"  Saved → {path}")


@torch.no_grad()
def evaluate(model, val_loader, device, fp16, max_batches=100):
    model.eval()
    totals = {"loss": 0, "itc_loss": 0, "itm_loss": 0, "itg_loss": 0}
    count  = 0
    for batch in val_loader:
        pv   = batch["pixel_values"].to(device)
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["labels"].to(device)
        with autocast("cuda", enabled=fp16):
            out = model(pv, ids, mask, lbl)
        for k in totals:
            totals[k] += out[k].item()
        count += 1
        if count >= max_batches:
            break
    model.train()
    return {k: v / max(count, 1) for k, v in totals.items()}


def train(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    args.fp16 = args.fp16 and device.type == "cuda"

    model, image_processor, tokenizer = build_stage1(
        vision_model     = args.vision_model,
        lm_model         = args.language_model,
        num_query_tokens = args.num_query_tokens,
    )
    model = model.to(device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader, val_loader = build_dataloaders(
        image_processor,
        tokenizer,
        stage             = 1,
        train_batch_size  = args.batch_size,
        eval_batch_size   = args.batch_size * 2,
        num_workers       = args.num_workers,
        seed              = args.seed,
    )
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.05,
    )
    total_steps   = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps  = int(total_steps * 0.05)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = GradScaler("cuda", enabled=args.fp16)

    output_dir    = Path(args.output_dir)
    latest        = output_dir / "latest.pt"
    start_epoch, global_step, best_val_loss = 0, 0, float("inf")

    if latest.exists():
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        global_step  = ckpt["step"]
        print(f"Resumed from epoch {start_epoch}")

    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        totals = {"loss": 0, "itc_loss": 0, "itm_loss": 0, "itg_loss": 0}
        t0     = time.time()

        for step, batch in enumerate(train_loader):
            pv   = batch["pixel_values"].to(device, non_blocking=True)
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            lbl  = batch["labels"].to(device, non_blocking=True)

            with autocast("cuda", enabled=args.fp16):
                out  = model(pv, ids, mask, lbl)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            for k in totals:
                totals[k] += out[k].item()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % args.log_steps == 0:
                    n = step + 1
                    print(f"Epoch {epoch+1} | Step {global_step} | "
                          f"loss={totals['loss']/n:.4f} "
                          f"ITC={totals['itc_loss']/n:.4f} "
                          f"ITM={totals['itm_loss']/n:.4f} "
                          f"ITG={totals['itg_loss']/n:.4f} | "
                          f"{time.time()-t0:.0f}s")

        n = len(train_loader)
        print(f"\nEpoch {epoch+1} | train loss={totals['loss']/n:.4f} "
              f"ITC={totals['itc_loss']/n:.4f} ITM={totals['itm_loss']/n:.4f} "
              f"ITG={totals['itg_loss']/n:.4f}")

        val = evaluate(model, val_loader, device, args.fp16)
        print(f"  val loss={val['loss']:.4f} ITC={val['itc_loss']:.4f} "
              f"ITM={val['itm_loss']:.4f} ITG={val['itg_loss']:.4f}")

        is_best = val["loss"] < best_val_loss
        if is_best:
            best_val_loss = val["loss"]

        save_checkpoint(model, optimizer, scaler, epoch, global_step, latest)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scaler, epoch, global_step,
                            output_dir / f"epoch_{epoch+1}.pt", is_best=is_best)

    save_checkpoint(model, optimizer, scaler, args.epochs - 1, global_step,
                    output_dir / "final.pt")
    print("Stage 1 complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vision_model",    default="openai/clip-vit-large-patch14")
    p.add_argument("--language_model",  default="google/flan-t5-base")
    p.add_argument("--num_query_tokens",type=int, default=32)
    p.add_argument("--output_dir",      default="./checkpoints/qformer_stage1")
    p.add_argument("--epochs",          type=int, default=20)
    p.add_argument("--batch_size",      type=int, default=32)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--grad_accum",      type=int, default=2)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--log_steps",       type=int, default=50)
    p.add_argument("--save_every",      type=int, default=2)
    p.add_argument("--fp16",            action="store_true", default=True)
    p.add_argument("--no_fp16",         dest="fp16", action="store_false")
    p.add_argument("--seed",            type=int, default=42)
    train(p.parse_args())
