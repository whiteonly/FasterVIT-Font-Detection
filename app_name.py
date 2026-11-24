"""Small desktop app to test the 'azman in Esthoria' detector (CPU, no extra installs).

Put this file in your fontft folder (next to model_utils.py, synth.py, name_data.py) and run:
    python app_name.py
Edit the three DEFAULT_ paths below once, or pass --checkpoint / --head / --font.

Buttons:
  Open image...        pick a picture from disk
  Paste image          use an image from the clipboard (e.g. a screenshot)
  Test word, Esthoria  draw the typed word in the Esthoria font and test it
  Test word, other     draw the typed word in a random other font and test it
"""
import argparse
import os
import queue
import random
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import torch
import torch.nn as nn
from PIL import Image, ImageGrab, ImageOps, ImageTk

from model_utils import build_model, to_tensor
from name_data import collect_fonts
from synth import render_word

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = r"C:\Users\Lenovo\Downloads\fastervit_4_21k_768_w48.pth"
DEFAULT_HEAD = next((p for p in (os.path.join(HERE, "head.pt"), os.path.join(HERE, "runs", "azman_cpu", "head.pt"))
                     if os.path.exists(p)), os.path.join(HERE, "head.pt"))
DEFAULT_FONT = r"C:\Users\Lenovo\Downloads\esthoria-demo\EsthoriaDemo-24Point.otf"


class App:
    def __init__(self, root, a):
        self.root, self.a = root, a
        self.q = queue.Queue()
        self.model = self.head = None
        self.photo = None
        win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        self.others = [f for f in collect_fonts(win_fonts)
                       if os.path.basename(f).lower() != os.path.basename(a.font).lower()]
        root.title("azman detector (Esthoria)")

        top = tk.Frame(root)
        top.pack(padx=10, pady=8, fill="x")
        self.btns = [
            tk.Button(top, text="Open image...", command=self.open_image),
            tk.Button(top, text="Paste image", command=self.paste_image),
        ]
        for b in self.btns:
            b.pack(side="left", padx=3)

        row = tk.Frame(root)
        row.pack(padx=10, pady=2, fill="x")
        tk.Label(row, text="Word:").pack(side="left")
        self.word = tk.Entry(row, width=14)
        self.word.insert(0, "azman")
        self.word.pack(side="left", padx=4)
        self.btns += [
            tk.Button(row, text="Test word, Esthoria", command=lambda: self.test_word(True)),
            tk.Button(row, text="Test word, other font", command=lambda: self.test_word(False)),
        ]
        self.btns[-2].pack(side="left", padx=3)
        self.btns[-1].pack(side="left", padx=3)
        tk.Label(row, text="  Threshold:").pack(side="left")
        self.thr = tk.Entry(row, width=5)
        self.thr.insert(0, "0.5")
        self.thr.pack(side="left")

        self.canvas = tk.Label(root, width=48, height=24, bg="#dddddd", text="(no image yet)")
        self.canvas.pack(padx=10, pady=8)
        self.result = tk.Label(root, text="", font=("Segoe UI", 18, "bold"))
        self.result.pack(pady=2)
        self.status = tk.Label(root, text="Loading model... (the 8 GB file takes a few minutes the first time)",
                               fg="#555555", wraplength=520)
        self.status.pack(padx=10, pady=(0, 10))

        self.set_busy(True)
        threading.Thread(target=self.load_model, daemon=True).start()
        root.after(150, self.poll)

    # ---- background work (never touch tk widgets from these threads; use the queue) ----
    def load_model(self):
        try:
            torch.set_num_threads(os.cpu_count() or 4)
            self.head = torch.load(self.a.head, map_location="cpu", weights_only=False)  # your own head.pt
            m = build_model(self.head["model_name"], 2, self.a.checkpoint)
            m.head = nn.Identity()
            self.model = m.eval()
            self.q.put(("ready", None))
        except BaseException as e:  # show any problem in the window
            self.q.put(("error", f"{type(e).__name__}: {e}"))

    def run_predict(self, img):
        try:
            size, h = self.head["img_size"], self.head
            if img.size != (size, size):
                img = ImageOps.pad(img, (size, size), color=img.getpixel((0, 0)))
            with torch.inference_mode():
                f = self.model(to_tensor(img)[None]).float()
                prob = ((f - h["mean"]) / h["std"] @ h["W"].T + h["b"]).softmax(1)[0, 1].item()
            self.q.put(("result", prob))
        except BaseException as e:
            self.q.put(("error", f"{type(e).__name__}: {e}"))

    # ---- UI side ----
    def poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "ready":
                    self.status.config(text=f"Model ready. Target word: '{self.head['target']}'. Pick an image or type a word.")
                    self.set_busy(False)
                elif kind == "result":
                    try:
                        thr = float(self.thr.get())
                    except ValueError:
                        thr = 0.5
                    yes = val >= thr
                    self.result.config(text=f"YES - {self.head['target']}  ({val:.0%})" if yes else f"no  ({val:.0%})",
                                       fg="#1a8f3c" if yes else "#c0392b")
                    self.status.config(text=f"P({self.head['target']}) = {val:.3f}, threshold {thr}")
                    self.set_busy(False)
                elif kind == "error":
                    self.status.config(text="Error: " + val, fg="#c0392b")
                    for b in self.btns:  # keep buttons off if the model never loaded
                        b.config(state="normal" if self.model is not None else "disabled")
        except queue.Empty:
            pass
        self.root.after(150, self.poll)

    def set_busy(self, busy):
        for b in self.btns:
            b.config(state="disabled" if busy else "normal")

    def show(self, img):
        view = img.copy()
        view.thumbnail((380, 380))
        self.photo = ImageTk.PhotoImage(view)
        self.canvas.config(image=self.photo, text="", width=view.width, height=view.height)
        self.result.config(text="working...", fg="#555555")
        self.status.config(text="Running the model on CPU - a few seconds...", fg="#555555")
        self.set_busy(True)
        threading.Thread(target=self.run_predict, args=(img.convert("RGB"),), daemon=True).start()

    def open_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")])
        if p:
            self.show(Image.open(p).convert("RGB"))

    def paste_image(self):
        img = ImageGrab.grabclipboard()
        if isinstance(img, Image.Image):
            self.show(img.convert("RGB"))
        else:
            messagebox.showinfo("Paste image", "No image on the clipboard. Copy or screenshot one first.")

    def test_word(self, esthoria):
        w = self.word.get().strip()
        if not w:
            return
        if esthoria:
            font = self.a.font
        elif self.others:
            font = random.choice(self.others)
        else:
            messagebox.showinfo("Other font", "No other fonts found on this PC.")
            return
        self.show(render_word(w, font, self.head["img_size"], random.Random(), augment=True))
        self.status.config(text=f"Drew '{w}' in {os.path.basename(font)} - running the model...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--head", default=DEFAULT_HEAD)
    ap.add_argument("--font", default=DEFAULT_FONT)
    a = ap.parse_args()
    for label, p in (("head.pt", a.head), ("checkpoint", a.checkpoint), ("font", a.font)):
        if not os.path.exists(p):
            raise SystemExit(f"{label} not found: {p}\nEdit the DEFAULT_ paths at the top of app_name.py or pass it as an argument.")
    root = tk.Tk()
    App(root, a)
    root.mainloop()


if __name__ == "__main__":
    main()