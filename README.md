# FasterViT Font Detection

Experiments and a small desktop app for recognizing words rendered in a specific font. The current example is a binary detector for **"azman" in Esthoria Demo**.

The project also includes scripts for training a multi-class word classifier with synthetic images.

## What is included

- **Desktop tester** - open or paste an image, or render a typed word in the configured font.
- **One-name detector** - classify an image as the target name in the target font or "other."
- **Multi-class detector** - train a word classifier using generated or on-the-fly images.
- **Synthetic image tools** - create varied word images for training and inspection.

## Requirements

- Python 3
- Packages listed in [`requirements.txt`](requirements.txt)
- A FasterViT-4-21K-768 pretrained checkpoint
- A `.ttf` or `.otf` font file

For GPU training, install the PyTorch build that matches your CUDA setup by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/). CPU inference is supported, but training and inference can be very slow.

## Setup

```powershell
git clone https://github.com/whiteonly/FasterVIT-Font-Detection.git
cd FasterVIT-Font-Detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the **FasterViT-4-21K-768** pretrained checkpoint from the [NVIDIA FasterViT model page](https://huggingface.co/nvidia/FasterViT). The checkpoint is a large external file and is not included in this repository.

The example font is [Esthoria Demo](https://www.1001fonts.com/esthoria-demo-font.html). Install it on your computer or pass the font file path directly.

## Run the desktop app

Pass paths to the pretrained checkpoint, app detector head, and target font:

```powershell
python app_name.py `
  --checkpoint "C:\path\to\fastervit_checkpoint.pth" `
  --head "runs\azman_cpu\head.pt" `
  --font "C:\path\to\EsthoriaDemo.otf"
```

The app also accepts these values through the `DEFAULT_CHECKPOINT`, `DEFAULT_HEAD`, and `DEFAULT_FONT` settings near the top of `app_name.py`. The default checkpoint and font paths are machine-specific, so update them or use the command-line options above.

The app's `--head` file is a separate binary-classifier artifact. It is not the same format as the `best.pth` checkpoint saved by `train_name.py`.

## Training workflows

### One-name detector

Optionally generate example images to inspect:

```powershell
python name_data.py `
  --font "C:\path\to\EsthoriaDemo.otf" `
  --target azman `
  --out data_azman `
  --n 400
```

Train the one-name model:

```powershell
python train_name.py `
  --checkpoint "C:\path\to\fastervit_checkpoint.pth" `
  --font "C:\path\to\EsthoriaDemo.otf" `
  --target azman `
  --out runs\azman
```

`train_name.py` renders training examples on the fly and saves `best.pth` and `last.pth` under the output directory. Use `predict_name.py` to run a saved training checkpoint:

```powershell
python predict_name.py --model runs\azman\best.pth --images test.png
```

### Multi-class word detector

Generate a dataset on disk, then train with it:

```powershell
python gen_data.py --font "C:\path\to\font.ttf" --words words.txt --out data
python train.py --checkpoint "C:\path\to\fastervit_checkpoint.pth" --data-dir data
```

Alternatively, train using on-the-fly images by providing `--font` and optionally `--words` to `train.py`. Use `predict.py` to run a saved multi-class checkpoint:

```powershell
python predict.py --model runs\font_ft\best.pth --images test.png
```

## Repository structure

```text
FasterVIT-Font-Detection/
├── README.md                 # Project overview and usage
├── requirements.txt          # Python dependencies
├── app_name.py               # Desktop tester for the one-name detector
├── model_utils.py            # Model loading and image preprocessing
├── name_data.py              # One-name synthetic sample generation
├── train_name.py             # One-name detector training
├── predict_name.py           # One-name detector inference
├── synth.py                  # Shared synthetic word-image renderer
├── gen_data.py               # Multi-class dataset generation
├── train.py                  # Multi-class detector training
├── predict.py                # Multi-class detector inference
├── words.txt                 # Example word list
├── sample_preview.png        # Example preview image
├── data_azman/               # Example images in other/ and target/ folders
├── runs/
│   └── azman_cpu/
│       └── head.pt           # App's binary detector head, if included
└── validate/
    └── Arial24.png           # Example validation image
```

Training commands write checkpoints under `runs/` by default. The pretrained FasterViT checkpoint must be downloaded separately.

## References

- [FasterViT pretrained model](https://huggingface.co/nvidia/FasterViT)
- [Kaggle FasterViT font-detection notebook](https://www.kaggle.com/code/asnawiarifin/fastervit-font-detection)
- [Esthoria Demo font](https://www.1001fonts.com/esthoria-demo-font.html)

## Local files

`font_location.txt` may contain a machine-specific font path; it is not read by the application and should only be shared if appropriate. Python's `__pycache__/` directory is generated automatically and does not need to be committed.

## Change the path first/ or use AI to locate your model and font application

`
**DEFAULT_CHECKPOINT**
**DEFAULT_FONT**
`

Current Program using windows Font that already register inside window computer.