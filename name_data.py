"""Synthetic data for a ONE-NAME detector (e.g. 'azman' written in the Esthoria Demo font).

Label 1 = the target name in the target font.
Label 0 = everything else: other names/words, near-miss spellings (azmin, azam, aziman ...),
          random letters, empty backgrounds, and (optionally) the target name in OTHER fonts.

Dump pictures to disk:
  python name_data.py --font C:\\path\\EsthoriaDemo.otf --target azman --out data_azman --n 400
"""
import argparse
import random
import string
from pathlib import Path

from PIL import Image

from synth import render_blank, render_word

NEG_WORDS = (
    "azmin azlan aiman amin ahmad aziz hamid zaman salman norman human woman amman azura hazman azwan azri faiz "
    "zul ali abu siti nur hassan omar john maria hello world text design sample font name the and man azam "
    "ammar arman azim izzat azmi zainal razman noman roman"
).split()


def mutate(word, rng):
    """A near-miss of `word` (one edit), never equal to it."""
    for _ in range(20):
        w, op = list(word), rng.choice(["sub", "del", "ins", "swap"])
        i = rng.randrange(len(w))
        if op == "sub":
            w[i] = rng.choice(string.ascii_lowercase)
        elif op == "del" and len(w) > 2:
            del w[i]
        elif op == "ins":
            w.insert(i, rng.choice(string.ascii_lowercase))
        elif op == "swap" and len(w) > 1:
            j = min(i + 1, len(w) - 1)
            w[i], w[j] = w[j], w[i]
        out = "".join(w)
        if out != word:
            return out
    return word + "x"


def styled(w, style):
    return (w.lower(), w.capitalize(), w.upper())[style]


def make_sample(rng, target, font, size=768, neg_fonts=(), cases=True):
    """Return (PIL image, label). ~50 % positives."""
    style = rng.randrange(3) if cases else 0
    if rng.random() < 0.5:
        return render_word(styled(target, style), font, size, rng), 1
    r = rng.random()
    if neg_fonts and r < 0.15:  # hard negative: the right name, wrong font
        return render_word(styled(target, style), rng.choice(neg_fonts), size, rng), 0
    r = rng.random()
    if r < 0.30:
        w = rng.choice([x for x in NEG_WORDS if x != target.lower()])
    elif r < 0.65:
        w = mutate(target.lower(), rng)
    elif r < 0.85:
        w = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
    else:
        return render_blank(size, rng), 0
    f = rng.choice(list(neg_fonts) + [font] * 3) if neg_fonts else font  # mostly the target font
    return render_word(styled(w, style), f, size, rng), 0


def collect_fonts(folder):
    if not folder:
        return []
    return [str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in (".ttf", ".otf")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True)
    ap.add_argument("--target", default="azman")
    ap.add_argument("--neg-fonts-dir", default=None, help="folder of other .ttf/.otf fonts (optional)")
    ap.add_argument("--no-case-variants", action="store_true", help="only the exact lowercase spelling")
    ap.add_argument("--out", default="data_azman")
    ap.add_argument("--n", type=int, default=400, help="number of pictures")
    ap.add_argument("--img-size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng, nf = random.Random(a.seed), collect_fonts(a.neg_fonts_dir)
    out = Path(a.out)
    for lab in ("target", "other"):
        (out / lab).mkdir(parents=True, exist_ok=True)
    thumbs = []
    for i in range(a.n):
        img, y = make_sample(rng, a.target, a.font, a.img_size, nf, not a.no_case_variants)
        img.save(out / ("target" if y else "other") / f"{i:05d}.jpg", quality=95)
        if len(thumbs) < 12:
            thumbs.append(img.resize((256, 256)))
    sheet = Image.new("RGB", (4 * 256, 3 * 256), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % 4) * 256, (i // 4) * 256))
    sheet.save(out / "preview.png")
    print(f"{a.n} pictures -> {out}/target and {out}/other  (preview.png)")


if __name__ == "__main__":
    main()
