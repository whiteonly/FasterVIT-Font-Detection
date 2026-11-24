"""python predict.py --model runs/font_ft/best.pth --images a.png b.jpg [--topk 3]"""
import argparse

import torch
from PIL import Image, ImageOps

from model_utils import build_model
from train import to_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=3)
    a = ap.parse_args()

    ck = torch.load(a.model, map_location="cpu", weights_only=False)  # your own file from train.py
    classes, size = ck["classes"], ck["img_size"]
    model = build_model(ck["model_name"], len(classes))
    model.load_state_dict(ck["state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    for p in a.images:
        img = Image.open(p).convert("RGB")
        if img.size != (size, size):  # letterbox to the training canvas, padding with the corner colour
            img = ImageOps.pad(img, (size, size), color=img.getpixel((0, 0)))
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            prob = model(to_tensor(img)[None].to(dev)).float().softmax(1)[0]
        top = prob.topk(min(a.topk, len(classes)))
        print(p, "->", ", ".join(f"{classes[i]} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices.tolist())))


if __name__ == "__main__":
    main()
