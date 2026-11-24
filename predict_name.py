"""python predict_name.py --model runs\\azman\\best.pth --images a.png b.jpg [--threshold 0.5]"""
import argparse

import torch
from PIL import Image, ImageOps

from model_utils import build_model
from train import to_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="raise it for fewer false alarms")
    a = ap.parse_args()

    ck = torch.load(a.model, map_location="cpu", weights_only=False)  # your own file from train_name.py
    size = ck["img_size"]
    model = build_model(ck["model_name"], 2)
    model.load_state_dict(ck["state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    for p in a.images:
        img = Image.open(p).convert("RGB")
        if img.size != (size, size):  # letterbox to the training canvas, padding with the corner colour
            img = ImageOps.pad(img, (size, size), color=img.getpixel((0, 0)))
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            prob = model(to_tensor(img)[None].to(dev)).float().softmax(1)[0, 1].item()
        verdict = f"YES, '{ck['target']}'" if prob >= a.threshold else "no"
        print(f"{p}: P({ck['target']}) = {prob:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
