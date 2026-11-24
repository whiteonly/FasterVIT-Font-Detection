"""Fine-tune FasterViT-4-21K-768 as a binary detector: is this the name 'azman' in the target font?

python train_name.py --checkpoint C:\\Users\\Lenovo\\Downloads\\fastervit_4_21k_768_w48.pth ^
       --font C:\\path\\EsthoriaDemo.otf --target azman [--neg-fonts-dir C:\\Windows\\Fonts]
Pictures are rendered on the fly (fresh every step), so nothing needs to be stored.
"""
import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_utils import build_model
from name_data import collect_fonts, make_sample
from train import to_tensor

CLASSES = ["other", "target"]


class OnlineNames(Dataset):
    def __init__(self, target, font, size, length, neg_fonts, cases, fixed_seed=None):
        self.a = (target, font, size, tuple(neg_fonts), cases)
        self.length, self.fixed_seed = length, fixed_seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = random.Random(os.urandom(8)) if self.fixed_seed is None else random.Random(self.fixed_seed * 1_000_003 + i)
        img, y = make_sample(rng, *self.a)
        return to_tensor(img), y


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    tp = fp = fn = tn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            p = model(x).argmax(1)
        tp += ((p == 1) & (y == 1)).sum().item()
        fp += ((p == 1) & (y == 0)).sum().item()
        fn += ((p == 0) & (y == 1)).sum().item()
        tn += ((p == 0) & (y == 0)).sum().item()
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return acc, tp / max(tp + fp, 1), tp / max(tp + fn, 1)  # accuracy, precision, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--font", required=True, help="Esthoria Demo .ttf/.otf file")
    ap.add_argument("--target", default="azman")
    ap.add_argument("--neg-fonts-dir", default=None, help="other fonts: teaches 'azman in another font = NOT target'")
    ap.add_argument("--no-case-variants", action="store_true")
    ap.add_argument("--model-name", default="faster_vit_4_21k_768")
    ap.add_argument("--img-size", type=int, default=768)
    ap.add_argument("--out", default="runs/azman")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=150, help="optimizer steps per epoch")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=2e-5)
    ap.add_argument("--freeze-epochs", type=int, default=1)
    ap.add_argument("--val-size", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device.type == "cpu":
        print("[WARN] no GPU found - this model is extremely slow on CPU.")

    nf, cases = collect_fonts(a.neg_fonts_dir), not a.no_case_variants
    print(f"target='{a.target}'  other fonts for negatives: {len(nf)}")
    train_ds = OnlineNames(a.target, a.font, a.img_size, a.steps_per_epoch * a.batch_size * a.accum, nf, cases)
    val_ds = OnlineNames(a.target, a.font, a.img_size, a.val_size, nf, cases, fixed_seed=1234)
    kw = dict(num_workers=a.workers, pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, a.batch_size, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, a.batch_size, **kw)

    model = build_model(a.model_name, 2, a.checkpoint).to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.backbone_lr}, {"params": head, "lr": a.head_lr}],
                            weight_decay=0.05)
    spe = len(train_dl) // a.accum
    total, warm = spe * a.epochs, max(1, spe // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
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
        acc, prec, rec = evaluate(model, val_dl, device, amp_dtype)
        print(f"epoch {ep + 1}/{a.epochs}  loss {run / cnt:.4f}  val acc {acc:.3f}  precision {prec:.3f}  "
              f"recall {rec:.3f}  ({time.time() - t0:.0f}s)")
        state = {"state_dict": model.state_dict(), "classes": CLASSES, "target": a.target,
                 "model_name": a.model_name, "img_size": a.img_size, "val_acc": acc}
        torch.save(state, out / "last.pth")
        if acc >= best:
            best = acc
            torch.save(state, out / "best.pth")
    print(f"done. best val acc {best:.3f} -> {out / 'best.pth'}")


if __name__ == "__main__":
    main()
