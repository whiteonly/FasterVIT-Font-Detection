"""Synthetic word-image renderer for a single, specific font.

Used by gen_data.py (writes images to disk) and train.py (renders on the fly).
Only needs Pillow + numpy.
"""
import random
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEFAULT_WORDS = (
    "the be to of and a in that have it for not on with he as you do at this but his by from they we say her she "
    "or an will my one all would there their what so up out if about who get which go me when make can like time no "
    "just him know take people into year your good some could them see other than then now look only come its over "
    "think also back after use two how our work first well way even new want because any these give day most us "
    "hello world font sample text design"
).split()


def load_words(path=None):
    """One word per line (or whitespace separated). Duplicates removed, order kept."""
    if path is None:
        words = DEFAULT_WORDS
    else:
        with open(path, "r", encoding="utf-8") as f:
            words = f.read().split()
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


@lru_cache(maxsize=256)
def _font(path, size):
    return ImageFont.truetype(path, size)


def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _colors(rng):
    """Random background / text colours with enough contrast."""
    for _ in range(30):
        bg = tuple(rng.randint(0, 255) for _ in range(3))
        fg = tuple(rng.randint(0, 255) for _ in range(3))
        if abs(_lum(bg) - _lum(fg)) > 120:
            return bg, fg
    return ((255, 255, 255), (0, 0, 0)) if rng.random() < 0.5 else ((0, 0, 0), (255, 255, 255))


def _background(size, bg, rng):
    kind = rng.choice(["solid", "gradient", "noise", "lines"])
    base = np.zeros((size, size, 3), np.float32) + np.array(bg, np.float32)
    if kind == "gradient":
        other = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
        other = 0.75 * np.array(bg, np.float32) + 0.25 * other
        t = np.linspace(0, 1, size, dtype=np.float32)
        t = t[None, :, None] if rng.random() < 0.5 else t[:, None, None]
        base = base * (1 - t) + other * t
    elif kind == "noise":
        base += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(4, 18), (size, size, 1))
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    if kind == "lines":
        d = ImageDraw.Draw(img)
        shade = tuple(int(np.clip(v + rng.randint(-25, 25), 0, 255)) for v in bg)
        for _ in range(rng.randint(2, 10)):
            d.line([(rng.randint(0, size), rng.randint(0, size)), (rng.randint(0, size), rng.randint(0, size))],
                   fill=shade, width=rng.randint(1, 4))
    return img


def render_blank(size=768, rng=None):
    """Background only (no text) - a 'nothing here' negative example."""
    rng = rng or random.Random()
    bg, _ = _colors(rng)
    return _background(size, bg, rng)


def render_word(word, font_path, size=768, rng=None, augment=True):
    """Return an RGB PIL image (size x size) with `word` drawn in `font_path`."""
    rng = rng or random.Random()
    # 1) choose a font size so the word spans ~45-90 % of the canvas width
    probe = _font(font_path, 100)
    l, t, r, b = probe.getbbox(word)
    w0, h0 = max(r - l, 1), max(b - t, 1)
    frac = rng.uniform(0.45, 0.90) if augment else 0.70
    fsize = int(100 * frac * size / w0)
    fsize = max(12, min(fsize, int(100 * 0.5 * size / h0)))
    font = _font(font_path, fsize)
    l, t, r, b = font.getbbox(word)
    pad = max(8, fsize // 4)

    # 2) draw the text on its own mask so it can be rotated / sheared / blurred
    mask = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(mask).text((pad - l, pad - t), word, font=font, fill=255)
    if augment:
        shear = rng.uniform(-0.08, 0.08)  # keep small: heavy shear looks like a different (italic) font
        mask = mask.transform(mask.size, Image.AFFINE, (1, shear, -shear * mask.size[1] / 2, 0, 1, 0),
                              resample=Image.BICUBIC)
        mask = mask.rotate(rng.uniform(-6, 6), resample=Image.BICUBIC, expand=True)
        if rng.random() < 0.4:
            mask = mask.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6)))

    # 3) compose on a background
    if augment:
        bg, fg = _colors(rng)
        canvas = _background(size, bg, rng)
    else:
        canvas, fg = Image.new("RGB", (size, size), (255, 255, 255)), (0, 0, 0)
    mw, mh = mask.size
    if mw > size or mh > size:  # safety: shrink if rotation pushed it out
        s = min(size / mw, size / mh)
        mask = mask.resize((max(1, int(mw * s)), max(1, int(mh * s))), Image.BICUBIC)
        mw, mh = mask.size
    if augment:
        x, y = rng.randint(0, size - mw), rng.randint(0, size - mh)
    else:
        x, y = (size - mw) // 2, (size - mh) // 2
    canvas.paste(Image.new("RGB", mask.size, fg), (x, y), mask)

    # 4) camera-like degradations
    if augment:
        if rng.random() < 0.3:
            canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
        if rng.random() < 0.5:
            arr = np.asarray(canvas, np.float32)
            arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 10), arr.shape)
            canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return canvas
