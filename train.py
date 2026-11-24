"""Fine-tune FasterViT-4-21K-768 so it recognises words written in ONE specific font.

Each word in the word list is a class. Images are rendered on the fly (infinite fresh augmentations)
or read from a folder made by gen_data.py (--data-dir).

python train.py --checkpoint C:\\Users\\Lenovo\\Downloads\\fastervit_4_21k_768_w48.pth --font C:\\path\\MyFont.ttf --words words.txt
"""
import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model_utils import build_model, to_tensor
from synth import load_words, render_word


class OnlineWords(Dataset):
    """Renders a fresh synthetic image every time (train) or a fixed one per index (val)."""

    def __init__(self, words, font, size, length, fixed_seed=None):
        self.words, self.font, self.size, self.length, self.fixed_seed = words, font, size, length, fixed_seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.fixed_seed is None:
            rng = random.Random(os.urandom(8))
            label = rng.randrange(len(self.words))
        else:
            rng = random.Random(self.fixed_seed * 1_000_003 + i)
            label = i % len(self.words)
        return to_tensor(render_word(self.words[label], self.font, self.size, rng)), label


class DiskWords(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        with open(self.root / f"{split}.csv", encoding="utf-8") as f:
            self.rows = [(r["path"], int(r["label"])) for r in csv.DictReader(f)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, y = self.rows[i]
        return to_tensor(Image.open(self.root / p)), y


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    ok = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            pred = model(x).argmax(1)
        ok += (pred == y).sum().item()
        n += y.numel()
    return ok / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-name", default="faster_vit_4_21k_768")
    ap.add_argument("--font", help="font file (needed for online rendering)")
    ap.add_argument("--words", default=None)
    ap.add_argument("--data-dir", default=None, help="folder from gen_data.py; omit to render online")
    ap.add_argument("--img-size", type=int, default=768)
    ap.add_argument("--out", default="runs/font_ft")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps-per-epoch", type=int, default=300, help="optimizer steps per epoch (online mode)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation (effective batch = bs*accum)")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=2e-5)
    ap.add_argument("--freeze-epochs", type=int, default=1, help="train only the new head for the first N epochs")
    ap.add_argument("--val-size", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device.type == "cpu":
        print("[WARN] no GPU found - a 768px, 272M-param model will be extremely slow on CPU.")

    if a.data_dir:
        classes = Path(a.data_dir, "classes.txt").read_text(encoding="utf-8").split("\n")
        train_ds, val_ds = DiskWords(a.data_dir, "train"), DiskWords(a.data_dir, "val")
        eff_len = len(train_ds)
    else:
        if not a.font:
            raise SystemExit("--font is required unless --data-dir is given")
        classes = load_words(a.words)
        eff_len = a.steps_per_epoch * a.batch_size * a.accum
        train_ds = OnlineWords(classes, a.font, a.img_size, eff_len)
        val_ds = OnlineWords(classes, a.font, a.img_size, max(a.val_size, len(classes)), fixed_seed=1234)
    print(f"{len(classes)} classes, {eff_len} train samples/epoch")

    kw = dict(num_workers=a.workers, pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, a.batch_size, shuffle=bool(a.data_dir), drop_last=True, **kw)
    val_dl = DataLoader(val_ds, a.batch_size, shuffle=False, **kw)

    model = build_model(a.model_name, len(classes), a.checkpoint).to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.backbone_lr}, {"params": head, "lr": a.head_lr}],
                            weight_decay=0.05)
    steps_per_epoch = len(train_dl) // a.accum
    total, warm = steps_per_epoch * a.epochs, max(1, steps_per_epoch // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(a.epochs):
        for p in body:
            p.requires_grad_(ep >= a.freeze_epochs)
        model.train()
        t0, run, cnt = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss / a.accum).backward()
            run, cnt = run + loss.item(), cnt + 1
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
        acc = evaluate(model, val_dl, device, amp_dtype)
        print(f"epoch {ep + 1}/{a.epochs}  loss {run / cnt:.4f}  val_acc {acc:.4f}  ({time.time() - t0:.0f}s)")
        state = {"state_dict": model.state_dict(), "classes": classes, "model_name": a.model_name,
                 "img_size": a.img_size, "val_acc": acc, "font": os.path.basename(a.font or "")}
        torch.save(state, out / "last.pth")
        if acc >= best:
            best = acc
            torch.save(state, out / "best.pth")
    print(f"done. best val_acc = {best:.4f}  -> {out / 'best.pth'}")


if __name__ == "__main__":
    main()
