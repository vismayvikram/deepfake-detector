# clip_baseline.py
# frozen CLIP ViT-B-32 + plain logistic regression probe
# just seeing how a baseline I didn't train at all does compared to mine, kind of a sanity check

import csv
import os
import open_clip
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

from data import get_dff_data, get_eval_transform, get_stylegan_data

RESULTS_FILE = "results.csv"
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def extract_features(clip_model, data_loader, device):
  feats, labels = [], []
  mean = CLIP_MEAN.view(1, 3, 1, 1).to(device)
  std = CLIP_STD.view(1, 3, 1, 1).to(device)

  with torch.no_grad():
    for x, y in data_loader:
      x = x.to(device)
      if x.shape[-2:] != (224, 224):
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bicubic", align_corners=False
        )
      x = (x - mean) / std
      f = torch.nn.functional.normalize(clip_model.encode_image(x), dim=-1)
      
      feats.append(f.cpu().numpy())
      labels.append(y.numpy())

  return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0).ravel()


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"running CLIP ViT-B-32 baseline on {device}...")

  # Load pretrained CLIP
  clip_model, _, _ = open_clip.create_model_and_transforms(
      "ViT-B-32", pretrained="openai"
  )
  clip_model = clip_model.to(device).eval()

  # Probe training data: ~3000 samples from StyleGAN train split
  print("extracting CLIP features for 3000 training samples...")
  train_loader = DataLoader(
      get_stylegan_data(
          split="train",
          max_per_class=1500,
          transform=get_eval_transform("clean"),
      ),
      batch_size=64,
      shuffle=False,
  )

  X_train, y_train = extract_features(clip_model, train_loader, device)

  print("fitting logistic regression probe on frozen features...")
  clf = LogisticRegression(max_iter=1000, C=1.0)
  clf.fit(X_train, y_train)

  # Evaluation sets and perturbations
  eval_sets = {
      "StyleGAN": lambda tf: get_stylegan_data(
          "test", max_per_class=1000, transform=tf
      ),
      "SD-Inpaint": lambda tf: get_dff_data(
          "inpainting", max_per_class=1000, transform=tf
      ),
      "InsightFace": lambda tf: get_dff_data(
          "insight", max_per_class=1000, transform=tf
      ),
      "SD-Text2Img": lambda tf: get_dff_data(
          "text2img", max_per_class=1000, transform=tf
      ),
  }
  perturbations = ["clean", "jpeg90", "jpeg60", "resize0.5", "blur"]

  rows = []
  for set_name, ds_fn in eval_sets.items():
    for pert in perturbations:
      loader = DataLoader(
          ds_fn(get_eval_transform(pert)), batch_size=64, shuffle=False
      )

      X_val, y_val = extract_features(clip_model, loader, device)
      preds_prob = clf.predict_proba(X_val)[:, 1]

      auc = roc_auc_score(y_val, preds_prob)
      acc = accuracy_score(y_val, (preds_prob >= 0.5).astype(int))

      print(
          f"clip_probe | {set_name:12s} | {pert:10s} | AUC: {auc:.4f} | Acc:"
          f" {acc:.4f}"
      )
      rows.append(
          ["clip_probe", set_name, pert, round(auc, 6), round(acc, 4)]
      )

  # Append rows to results.csv
  file_exists = os.path.exists(RESULTS_FILE)
  with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
      writer.writerow(["model", "set", "perturbation", "auc", "acc"])
    writer.writerows(rows)

  print(f"appended {len(rows)} rows to {RESULTS_FILE}")


if __name__ == "__main__":
  main()