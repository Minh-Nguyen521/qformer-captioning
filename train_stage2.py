"""
Stage 2: Vision-to-Language Generative Fine-tuning

Loads Q-Former weights from Stage 1, then trains the FC projection and
optionally the language model.

Usage:
    python train_stage2.py --stage1_checkpoint ./checkpoints/qformer_stage1/best.pt
    python train_stage2.py --stage1_checkpoint ./checkpoints/qformer_stage1/best.pt \
                           --unfreeze_lm --lr 5e-5
"""

import argparse
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import set_seed, get_cosine_schedule_with_warmup

from model import build_stage2
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
def evaluate(model, val_loader, device, fp16, tokenizer, max_batches=50):
    model.eval()
    total_loss, count = 0.0, 0
    for batch in val_loader:
        pv, inp, mask, lbl = (batch[k].to(device) for k in
                              ("pixel_values", "input_ids", "attention_mask", "labels"))
        with autocast("cuda", enabled=fp16):
            out = model(pv, inp, mask, lbl)
        total_loss += out.loss.item()
        count += 1
        if count >= max_batches:
            break

    # Sample captions
    batch = next(iter(val_loader))
    pv    = batch["pixel_values"][:4].to(device)
    caps  = model.generate(pv, tokenizer)
    print("\n  Sample captions:")
    for i, c in enumerate(caps):
        print(f"    [{i}] {c}")
    print()

    model.train()
    return total_loss / max(count, 1)


def train(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    args.fp16 = args.fp16 and device.type == "cuda"

    model, image_processor, tokenizer = build_stage2(
        vision_model     = args.vision_model,
        lm_model         = args.language_model,
        freeze_lm        = not args.unfreeze_lm,
        num_query_tokens = args.num_query_tokens,
    )

    if args.stage1_checkpoint and Path(args.stage1_checkpoint).exists():
        model.load_stage1_weights(args.stage1_checkpoint, device)
    else:
        print("[warn] No Stage 1 checkpoint found — training from scratch.")

    model = model.to(device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader, val_loader = build_dataloaders(
        image_processor,
        tokenizer,
        stage            = 2,
        train_batch_size = args.batch_size,
        eval_batch_size  = args.batch_size * 2,
        num_workers      = args.num_workers,
        seed             = args.seed,
    )
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # Separate LR: projection > Q-Former > LM
    proj_params = list(model.qformer_to_lm.parameters())
    proj_ids    = {id(p) for p in proj_params}
    lm_params   = [p for p in model.language_model.parameters() if p.requires_grad]
    qf_params   = [p for p in model.qformer.parameters()
                   if p.requires_grad and id(p) not in proj_ids]

    optimizer = AdamW([
        {"params": proj_params, "lr": args.lr * 5},
        {"params": lm_params,   "lr": args.lr},
        {"params": qf_params,   "lr": args.lr * 2},
    ], weight_decay=0.05)

    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * 0.05)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler("cuda", enabled=args.fp16)

    output_dir  = Path(args.output_dir)
    latest      = output_dir / "latest.pt"
    start_epoch, global_step, best_val_loss = 0, 0, float("inf")

    if latest.exists():
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        print(f"Resumed from epoch {start_epoch}")

    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        t0         = time.time()

        for step, batch in enumerate(train_loader):
            pv, inp, mask, lbl = (batch[k].to(device, non_blocking=True) for k in
                                  ("pixel_values", "input_ids", "attention_mask", "labels"))

            with autocast("cuda", enabled=args.fp16):
                out  = model(pv, inp, mask, lbl)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += out.loss.item()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % args.log_steps == 0:
                    avg = epoch_loss / (step + 1)
                    lr  = optimizer.param_groups[0]["lr"]
                    print(f"Epoch {epoch+1} | Step {global_step} | "
                          f"loss={avg:.4f} | lr={lr:.2e} | {time.time()-t0:.0f}s")

        print(f"\nEpoch {epoch+1} done — avg loss: {epoch_loss/len(train_loader):.4f}")

        val_loss = evaluate(model, val_loader, device, args.fp16, tokenizer)
        print(f"  val loss: {val_loss:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(model, optimizer, scaler, epoch, global_step, latest)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scaler, epoch, global_step,
                            output_dir / f"epoch_{epoch+1}.pt", is_best=is_best)

    save_checkpoint(model, optimizer, scaler, args.epochs - 1, global_step,
                    output_dir / "final.pt")
    print("Stage 2 complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vision_model",       default="openai/clip-vit-large-patch14")
    p.add_argument("--language_model",     default="google/flan-t5-base")
    p.add_argument("--stage1_checkpoint",  default="./checkpoints/qformer_stage1/best.pt")
    p.add_argument("--num_query_tokens",   type=int, default=32)
    p.add_argument("--unfreeze_lm",        action="store_true", default=False)
    p.add_argument("--output_dir",         default="./checkpoints/qformer_stage2")
    p.add_argument("--epochs",             type=int, default=10)
    p.add_argument("--batch_size",         type=int, default=16)
    p.add_argument("--lr",                 type=float, default=1e-5)
    p.add_argument("--grad_accum",         type=int, default=4)
    p.add_argument("--num_workers",        type=int, default=4)
    p.add_argument("--log_steps",          type=int, default=50)
    p.add_argument("--save_every",         type=int, default=1)
    p.add_argument("--fp16",               action="store_true", default=True)
    p.add_argument("--no_fp16",            dest="fp16", action="store_false")
    p.add_argument("--seed",               type=int, default=42)
    train(p.parse_args())
