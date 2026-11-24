"""Write a synthetic image dataset (word images in YOUR font) to disk.

python gen_data.py --font C:\\path\\MyFont.ttf --words words.txt --out data --train-per-word 30 --val-per-word 5
"""
import argparse
import csv
import random
from pathlib import Path

from PIL import Image

from synth import load_words, render_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True, help=".ttf / .otf file of the font to learn")
    ap.add_argument("--words", default=None, help="text file, one word per line (default: small built-in list)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--img-size", type=int, default=768, help="must equal the model input (768 for the 768 model)")
    ap.add_argument("--train-per-word", type=int, default=30)
    ap.add_argument("--val-per-word", type=int, default=5)
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg", help="jpg keeps the folder small")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    words = load_words(a.words)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(words), encoding="utf-8")
    rng = random.Random(a.seed)

    for split, n in (("train", a.train_per_word), ("val", a.val_per_word)):
        (out / split).mkdir(exist_ok=True)
        rows = []
        for ci, w in enumerate(words):
            for k in range(n):
                img = render_word(w, a.font, a.img_size, rng, augment=True)
                rel = f"{split}/{ci:04d}_{k:04d}.{a.format}"
                if a.format == "jpg":
                    img.save(out / rel, quality=95)
                else:
                    img.save(out / rel)
                rows.append((rel, ci))
        with open(out / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([("path", "label")] + rows)
        print(f"{split}: {len(rows)} images")

    # small contact sheet so you can eyeball the result
    sheet = Image.new("RGB", (4 * 256, 3 * 256), "white")
    for i in range(12):
        img = render_word(rng.choice(words), a.font, a.img_size, rng).resize((256, 256))
        sheet.paste(img, ((i % 4) * 256, (i // 4) * 256))
    sheet.save(out / "preview.png")
    print(f"{len(words)} classes -> {out}/ (see preview.png)")


if __name__ == "__main__":
    main()
