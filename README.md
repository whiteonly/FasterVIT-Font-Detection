# Detect the name "azman" in the Esthoria Demo font (FasterViT-4-21K-768)

Binary detector: YES = the word "azman" written in the Esthoria font, NO = anything else
(other names, near-miss spellings like azmin/azam, random letters, empty backgrounds,
and optionally "azman" in other fonts).

```
pip install -r requirements.txt          # install torch with CUDA first from pytorch.org

# 1) look at / save the synthetic pictures (target/ and other/ folders + preview.png)
python name_data.py --font C:\path\EsthoriaDemo.otf --target azman --out data_azman --n 400

# 2) train (pictures are rendered on the fly)
python train_name.py --checkpoint C:\Users\Lenovo\Downloads\fastervit_4_21k_768_w48.pth --font C:\path\EsthoriaDemo.otf --target azman --neg-fonts-dir C:\Windows\Fonts

# 3) test
python predict_name.py --model runs\azman\best.pth --images test.png
```
`--neg-fonts-dir` is optional: with it, "azman" in any other font counts as NO. Drop it if other fonts should also count.
Input is fixed at 768x768. Out of GPU memory: `--batch-size 1 --accum 16`.
(The older word-classifier scripts, train.py / gen_data.py, are still here but not needed for this task.)
"# FasterVIT-Font-Detection" 
