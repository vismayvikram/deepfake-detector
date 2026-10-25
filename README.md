# Multi-Modal Synthetic Image Detector

A portfolio/learning project exploring whether fusing spatial CNN features with DCT frequency-domain features generalizes better to unseen AI generators than a spatial-only baseline. Short answer: **not clearly**, at least not on this training setup. Details below.

---

## Research Question

Does a spatial + DCT-frequency fusion model generalize better to generator types it never saw during training, and to JPEG/resize/blur degradation, than a spatial-only ResNet-18 baseline?

Honest answer: **No, or not clearly.** All models collapsed to near-random (~0.5 AUC) on the three unseen generator sets. The fusion model was not meaningfully better than spatial-only on out-of-distribution generators. See the [Results](#results) section.

---

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `torch-dct`, `open_clip_torch`, `scikit-learn`, `streamlit`, `huggingface_hub`.

### Data

**Seen-generator training set** (StyleGAN / FFHQ):
```bash
kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw/stylegan --unzip
```
→ `data/raw/stylegan/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/`

**Unseen-generator test sets** (OpenRL/DeepFakeFace):  
Downloaded automatically in `data.py` via `huggingface_hub`. Four subsets:
- `wiki` — real images (IMDB-WIKI)
- `inpainting` — SD Inpainting fakes
- `insight` — InsightFace face-swap fakes
- `text2img` — SD 1.5 text-to-image fakes

> **Note on domain shift:** "real" in the StyleGAN split is FFHQ (studio-quality aligned faces). "Real" in the DFF split is IMDB-WIKI (noisy, in-the-wild photos). So bad unseen-generator scores could be partly real-image domain shift, not purely a failure to detect generator artifacts.

---

## Files

```
data.py           download helpers + Dataset classes + augmentation/perturbation transforms
model.py          Detector class (spatial + freq branches + fusion)
train.py          training loop — 6 checkpoints: {spatial,freq,fusion} x {no_aug,aug}
eval.py           evaluates all checkpoints on 4 sets x 5 perturbations, saves results.csv + plots
clip_baseline.py  frozen CLIP ViT-B-32 + logistic regression probe (sanity check baseline)
app.py            small Streamlit prototype app
explore.ipynb     messy first-pass exploration notebook (start here)
requirements.txt
.gitignore
```

No `src/` layout, no `__init__.py`, no pytest, no CI/CD. Scripts import each other directly.

---

## Model Architecture (`model.py`)

`Detector(mode)` — `mode` in `{"spatial", "freq", "fusion"}`:

- **Spatial branch:** pretrained ResNet-18 (fine-tuned) → `nn.Identity()` fc → 512-d features → `LayerNorm`
- **Freq branch:** grayscale → 2D DCT (`torch_dct.dct_2d`) → `log1p(|coefficients|)` → per-image normalize → 5-conv-block CNN from scratch → 128-d features → `LayerNorm`
  - DCT was used instead of FFT because FFT's periodic tiling assumption creates a bright cross artifact through the center of every spectrum (from edge discontinuities), which has nothing to do with generator fingerprints. DCT's symmetric tiling avoids that. Whether it actually helped numerically — see limitations below.
- **Fusion:** concatenate both normalized feature vectors (512+128=640) → small MLP head
- Each branch has its own auxiliary output head for direct gradient signal in fusion mode
- During training only, in fusion mode: randomly zero out one branch's features ~25% of the time (to prevent the model always leaning on spatial)
- Loss: `BCE(main_out) + 0.3 * BCE(spatial_aux) + 0.3 * BCE(freq_aux)` in fusion mode

---

## Training (`train.py`)

Trained on StyleGAN only (train split). The DFF unseen sets were never touched during training.

```bash
python train.py --mode spatial
python train.py --mode spatial --aug
python train.py --mode freq
python train.py --mode freq --aug
python train.py --mode fusion
python train.py --mode fusion --aug
```

AdamW + cosine LR schedule. Best val AUC checkpoint saved. Val = StyleGAN validation split only.

Produces 6 checkpoints in `checkpoints/`.

---

## Evaluation (`eval.py`)

```bash
python eval.py
```

Evaluates all 6 model checkpoints on **4 sets x 5 perturbation conditions** and saves `results.csv` + two plots.

```bash
python clip_baseline.py   # appends clip_probe rows to results.csv
```

---

## Results

### Seen Generator (StyleGAN) — Clean

All spatial/fusion models got >= 0.997 AUC and >= 98% accuracy on the seen generator. CLIP probe got 0.940 AUC.

| Model        |   AUC |   Acc |
|:-------------|------:|------:|
| spatial      | 0.999 | 0.984 |
| spatial_aug  | 0.998 | 0.975 |
| fusion       | 0.998 | 0.980 |
| fusion_aug   | 0.997 | 0.967 |
| freq         | 0.714 | 0.656 |
| freq_aug     | 0.662 | 0.617 |
| clip_probe   | 0.940 | 0.859 |

### Unseen Generators (clean condition) — The real question

| Model       | SD-Inpaint AUC | InsightFace AUC | SD-Text2Img AUC |
|:------------|---------------:|----------------:|----------------:|
| spatial     | 0.552          | 0.547           | 0.447           |
| spatial_aug | 0.544          | 0.535           | 0.525           |
| fusion      | 0.570          | 0.556           | 0.503           |
| fusion_aug  | 0.565          | 0.559           | 0.568           |
| freq        | 0.539          | 0.519           | 0.667           |
| freq_aug    | 0.570          | 0.553           | 0.687           |
| clip_probe  | 0.454          | 0.495           | 0.460           |

**Everything collapsed to near-random on unseen generators.** The gaps between models are in the 0.01–0.05 AUC range with only ~1000-image eval sets per condition — well within noise. There is no result here that reliably shows fusion generalizing better than spatial-only. The freq-only model actually had marginally higher AUC on SD-Text2Img (~0.67 vs ~0.50 for fusion), but the accuracy was still stuck at ~50%, meaning the ranking survived slightly but the 0.5 threshold completely failed.

### AUC vs Accuracy Divergence

Worth flagging explicitly: AUC ~0.55–0.68 while accuracy is ~0.49–0.51 across unseen generators. This means the models are ranking real vs fake at slightly-better-than-random, but the 0.5 threshold is essentially useless — the learned decision boundary from StyleGAN doesn't transfer. High AUC + ~50% accuracy = model has a slightly-useful ranking but a completely wrong threshold. Both metrics were reported for this reason.

### Perturbation / Degradation (StyleGAN, all models stay high)

On the seen generator, spatial/fusion models stayed above 0.97 AUC across all perturbations. Adding augmentation (`_aug` variants) helped slightly on JPEG-60 and resize-0.5 conditions. Full numbers are in `results.csv` and plotted in `degradations.png`.

---

## Limitations

- **Single training run, single seed per configuration.** Results could shift meaningfully with different seeds — treat these as a rough picture, not precise measurements.
- **Small unseen eval sets.** ~1000 images per generator type means a 0.03 AUC difference is noise, not a result.
- **Real-image domain shift.** FFHQ (training reals) vs IMDB-WIKI (DFF reals) differ substantially in quality, alignment, and background. It's impossible to tell how much of the ~0.5 AUC on unseen sets is from generator-artifact failure vs the model keying off real-image distribution shift.
- **Faces only, single dataset.** Trained on one GAN (StyleGAN2), evaluated on diffusion + face-swap. The training distribution is extremely narrow.
- **DCT vs FFT.** The DCT choice has a good theoretical reason (avoids the boundary-discontinuity cross artifact). But the actual unseen-generator numbers didn't improve over what a spatial-only baseline gets — so whether DCT-specific patterns transferred is unclear. Didn't run a matched FFT comparison to verify.
- **Freq branch is weak standalone.** The freq-only model peaked at 0.71 AUC on StyleGAN, well below spatial (~0.999). In fusion mode the spatial branch likely dominated — the feature dropout during training was intended to address this but clearly didn't fix it completely.

---

## CLIP Probe Baseline

A frozen CLIP ViT-B-32 (never fine-tuned) + sklearn LogisticRegression trained on ~3000 extracted features, evaluated the same way as the main models. On unseen generators it did no better than the trained models (AUC 0.42–0.50), suggesting this is a fundamental generalization problem, not something fixable by just using a stronger backbone.

---

## Streamlit App

```bash
streamlit run app.py
```

Loads `checkpoints/fusion_aug.pt`, shows a prediction score and DCT spectrum for uploaded images.

**Important:** this is a single-photo classifier, not a video deepfake detector. It cannot evaluate temporal tells (blinking irregularities, lip-sync drift) — that is a different and harder problem.

---

## Repo

Built over a few days as a learning project — messy code is intentional. See `explore.ipynb` for the initial exploration (real vs fake image counts, mean DCT spectrum comparison, quick prototype sanity check). Scripts are flat in the repo root, not a package.
