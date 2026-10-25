Build a multi-modal (spatial + frequency-domain) synthetic image detector.
Research question: does fusing spatial CNN features with DCT-domain features
generalize better to generator types the model never saw during training,
and to JPEG/resize/blur degradation, than a spatial-only baseline? Report
the honest answer, even if it's "no" or "not clearly."

=== STYLE: THIS SHOULD LOOK LIKE A STUDENT BUILT IT, NOT A COMPANY ===
This is a portfolio/learning project, not production software. It should
read like one person's own work over a few weeks, not AI-generated
boilerplate. Specifically:

- NO pytest/unit tests, NO GitHub Actions or any CI/CD, NO Docker, NO config
  frameworks (no Hydra/OmegaConf/dataclass-based settings — just plain
  variables at the top of each script, argparse is fine for basic --mode/
  --epochs style flags, nothing fancier than that), NO logging library (use
  print() everywhere), NO type hints on every function — a few here and
  there if they genuinely help is fine, but don't annotate everything.
- Simple variable names: img, x, y, df, lr, bs, ep, acc, preds — not
  elaborate descriptive names everywhere.
- Casual comments throughout, written like a student talking to themselves:
  "# trying a bigger kernel here", "# not sure if this actually helps but
  let's see", "# quick sanity check, ignore this", "# this is kinda hacky
  but it works", "# magic number, just went with what seemed to work",
  "# TODO clean this up later". Not every line needs a comment — just
  enough to feel like real working notes, not polished documentation.
- Progress tracking via print() and/or tqdm progress bars — not a logging
  module, not structured log output.
- Don't over-organize. A few scripts, not a deeply layered package.

IMPORTANT DISTINCTION: casual code style is wanted, but honest results are
still required. Messy code is fine. Misreported or cherry-picked results
are not — see the reporting section below, that part doesn't get relaxed.

=== DATA: REAL ONLY, NO SUBSTITUTES ===
Real photographs for "real," real output images from actual generative
models for "fake." Never procedurally generated placeholder images (no
hand-drawn gradients, no synthetic sine-wave "artifacts" injected in code
to fake a dataset) — that would make every result meaningless, no matter
how casual or polished the code around it is.

1. Train + seen-generator test: Kaggle "xhlulu/140k-real-and-fake-faces"
   (70k real FFHQ + 70k StyleGAN fake, 256px).
   kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw/stylegan --unzip
   -> data/raw/stylegan/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/

2. Unseen-generator test: Hugging Face "OpenRL/DeepFakeFace" — four zips
   (wiki=real/IMDB-WIKI, inpainting=SD Inpainting, insight=InsightFace
   face-swap, text2img=SD 1.5 text-to-image), ~1000 images each via
   huggingface_hub.hf_hub_download, into data/dff/{wiki,inpainting,insight,text2img}/.

Note somewhere (README or a comment) that "real" differs between the two
sources (FFHQ vs IMDB-WIKI), so bad unseen-generator scores could partly be
that domain shift, not purely a generator-artifact problem.

=== REPO: JUST A FEW FILES ===
data.py        - download helpers + Dataset class + transforms (jpeg/blur aug)
model.py       - the Detector class (spatial + freq branches + fusion)
train.py       - training loop, plain argparse for --mode/--epochs/--aug
eval.py        - runs every checkpoint on every set x perturbation, dumps a
                  results.csv, and makes a couple of matplotlib plots at the
                  bottom of the same file (no separate report.py needed)
clip_baseline.py - the comparison baseline (see below)
app.py         - small Streamlit app
explore.ipynb  - the messy first-pass notebook (see below)
requirements.txt (needs torch-dct in here), .gitignore (data/, checkpoints/, *.pt)

That's it. No src/ package layout needed, no __init__.py ceremony — just
these files sitting in the repo root, imported from each other directly.

=== NOTEBOOK FIRST, AND IT SHOULD ACTUALLY LOOK LIKE EXPLORATION ===
Build explore.ipynb before any of the scripts. This is where the real
figuring-out happens, so let it look like that:
- A few dead-end or "just checking" cells are fine (e.g. printing an image's
  shape before remembering to resize it, a quick "does this even load"
  check before building the real Dataset class).
- Comments like "# ok that's way too slow, trying a smaller batch",
  "# wait this looks wrong, let me check the labels", "# yeah that's better".
- Cover: load a few images and look at them, count real vs fake per split,
  plot a sample grid, then the mean-DCT-spectrum comparison (average
  log-magnitude DCT of real vs fake, plotted as real / fake / fake-minus-
  real panels, same panel layout we used for the FFT version but swap in
  DCT). Note in a markdown cell why DCT was picked over FFT here: FFT's
  periodic-tiling assumption creates a boundary-discontinuity artifact (a
  bright cross through the center of every spectrum, real or fake, just
  from image edges not wrapping smoothly) that has nothing to do with
  being a generator fingerprint — DCT's symmetric tiling avoids that, so
  whatever shows up in the fake-minus-real panel is more likely to be a
  real signal. Still worth sanity-checking by eye rather than assuming
  that fixes everything. Then build + sanity-check the model on a tiny
  subset, train a quick 1-epoch run just to see loss go down, and stop
  there — the real training run happens in train.py, not the notebook.
- The Detector class written in the notebook must be copied into model.py
  unchanged afterward (not rewritten), so a checkpoint saved from one works
  with the other.

=== MODEL (model.py) ===
One Detector class, mode in {"spatial","freq","fusion"}:
- spatial branch: pretrained ResNet-18 (fine-tuned, this is the transfer
  learning piece), fc swapped for nn.Identity, normalize with ImageNet
  mean/std -> 512-d features
- freq branch: grayscale -> 2D DCT (torch_dct.dct_2d) -> log1p(magnitude of
  coefficients) -> normalize per image -> small 5-conv-block CNN from
  scratch -> 128-d features (comment like "# using dct instead of fft,
  avoids that weird cross artifact fft spectrums always have" and "# no
  pretrained model does dct input so this one's starting from zero").
  Note: unlike FFT, don't apply any shift operation here — DCT already puts
  the low-frequency/DC component in the top-left corner by convention, so
  there's no fftshift-equivalent step needed.
- before concatenating: LayerNorm each branch's features (comment: "#
  without this the spatial branch's bigger numbers were probably just
  drowning out freq"), and give each branch its own small extra output head
  so it gets direct gradient signal even in fusion mode (comment: "# adding
  this so the freq branch doesn't just get ignored, trying it out")
- during training only, in fusion mode, randomly zero out one branch's
  features sometimes (~25% chance each) before the final layer so the model
  can't just always lean on spatial (comment: "# forcing it to sometimes
  work without spatial, see if freq branch actually learns something")
- forward() returns (main_out, spatial_out, freq_out); loss = BCE on
  main_out + ~0.3x BCE on each side output, in fusion mode

IMG_SIZE = 256 as a plain variable at the top of the file.

=== TRAINING (train.py) ===
Train only on StyleGAN (never touch the unseen sets during training — "#
these are for testing generalization, not training on"). Produce 6
checkpoints: {spatial, freq, fusion} x {no aug, with jpeg/blur aug}. Plain
AdamW + cosine schedule, print the loss/val numbers each epoch instead of
any fancy logging, save whichever epoch had the best val AUC (val = StyleGAN
validation split only, not the unseen sets — that'd be cheating).

Optional if there's time: a --freeze_freq flag that loads the already-
trained freq-only checkpoint, freezes just that branch, and trains the rest
("# pretrain freq on its own first since it was too weak to compete
otherwise").

=== EVAL (eval.py) ===
Run every checkpoint on 4 sets (StyleGAN=seen, SD-Inpaint, InsightFace,
SD-Text2Img=unseen) x 5 conditions (clean, jpeg90, jpeg60, resize0.5, blur),
print results as you go, save one results.csv (model, set, perturbation,
auc, acc columns), then a couple of quick matplotlib bar/line charts at the
bottom of the same script. Report BOTH auc and accuracy — they can diverge
(decent auc but ~50% accuracy means ranking kind of survived but the 0.5
threshold didn't), and that gap is itself worth noting if you see it.

=== CLIP BASELINE (clip_baseline.py) ===
A simple comparison: frozen CLIP ViT-B-32 (open_clip_torch, pretrained=
"openai", never fine-tuned) + a plain sklearn LogisticRegression trained on
~3000 extracted features. Evaluate the same way as eval.py, append rows to
the same results.csv as "clip_probe" ("# just seeing how a baseline I
didn't train at all does compared to mine, kind of a sanity check").

=== RESULTS / README — DON'T RELAX THIS PART ===
Report results exactly as they come out. If fusion doesn't clearly beat
spatial-only on unseen generators (if the gap is under ~0.03-0.05 AUC with
only a few hundred images per cell, that's noise, not a result — say so),
write that plainly along with your best guess why (branch dominance, freq
branch being much weaker standalone, small/single-generator training set)
instead of spinning a mixed result as a win. Don't show only the metric
that looks good if another metric for the same comparison tells a
different story. Also report honestly whether switching to DCT actually
helped vs what FFT would likely have given — don't just assume the
boundary-artifact argument for DCT translated into better numbers, check
it. Add a short, honest limitations note (single run, small eval sets, the
real-image source mismatch mentioned above, faces only).

=== STREAMLIT APP (app.py) ===
Small and simple — file uploader, load the best fusion_aug checkpoint using
the same Detector class from model.py (import it, don't redefine it),
show the prediction and maybe the DCT spectrum of the uploaded image.
Label it clearly as a prototype, not a real deepfake-video detector —
it's classifying single photos, not catching the frame-to-frame tells
(blinking, lip-sync drift) that video deepfakes often have, which is a
different and harder problem.

=== GIT ===
Commit as you go, casual messages are fine ("fixed the dct normalization
bug", "training loop works now", "adding the clip comparison"), doesn't
need to be a formal staged plan. Just make sure each message actually
describes what's in that commit — don't write "added real data" in a
commit if what's actually there is placeholder/synthetic data.