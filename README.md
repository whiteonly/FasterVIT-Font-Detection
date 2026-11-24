# Detect the name "azman" in the Esthoria Demo font (FasterViT-4-21K-768)

Binary detector: YES = the word "azman" written in the Esthoria font, NO = anything else
(other names, near-miss spellings like azmin/azam, random letters, empty backgrounds,
and optionally "azman" in other fonts).

# TRAIN THE MODEL LINK (CLOUD)
the model specification **FasterViT-4-21K-768**
[This is model pretrained](https://www.kaggle.com/code/asnawiarifin/fastervit-font-detection)

# TRAIN ON LOCALLY
Use the file stucture install here
**REMEMBER TO DOWNLOAD REQURIEMENT LIBARARY**
[REQUIREMENT](./requirements.txt)

# STEP BY STEP INSTALLATION
## download fasterVIT pretrained model

[This is model link](https://huggingface.co/nvidia/FasterViT)
- choose this model :**FasterViT-4-21K-768** and download the file

## Font installation 
In this case, i using **esthoria demo** font
[link](https://www.1001fonts.com/esthoria-demo-font.html)

## File documentation

```text
fontft/
├── README.md                 # Project overview, setup, and file documentation
├── requirements.txt          # Python package dependencies
├── app_name.py               # Desktop UI for testing images and typed words
├── model_utils.py            # Model creation, checkpoint loading, and image preprocessing
├── synth.py                  # Synthetic word-image rendering utilities
├── name_data.py              # Creates samples for the one-name binary detector
├── train_name.py             # Trains the one-name detector
├── predict_name.py           # Runs the one-name detector on image files
├── train.py                  # Trains a multi-class word/font detector
├── predict.py                # Runs the multi-class detector on image files
├── gen_data.py               # Generates an on-disk multi-class image dataset
├── words.txt                 # Word list used for training or data generation
├── sample_preview.png        # Example image
├── data_azman/               # Example one-name images, grouped by target and other
├── runs/
│   └── azman_cpu/            # model that been train on kaggle
│       └── head.pt           # Trained binary-detector head used by the desktop app
└── validate/                 # store the file upload on app_name.py
    └── Arial24.png           # Example validation image
```

`runs/azman_cpu/head.pt` contains the trained detector head. The pretrained
FasterViT checkpoint is a separate, much larger file and is not included in
this repository; provide its path with `--checkpoint` when running the app.

`font_location.txt`, if present, is a local note and is not read by the
application. Avoid committing it if it contains a machine-specific font path.
Python's `__pycache__/` directory is generated automatically and does not need
to be committed.

# UI APPLICATION RUNNING (REPO MODEL)
## Change the path first/ or use AI to locate your model and font application
`
**DEFAULT_CHECKPOINT**
**DEFAULT_FONT**
`
Current Program using windows Font that already register inside window computer.