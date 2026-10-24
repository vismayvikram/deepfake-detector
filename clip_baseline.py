# clip_baseline.py
# frozen CLIP ViT-B-32 + plain logistic regression probe
# just seeing how a baseline I didn't train at all does compared to mine, kind of a sanity check

import os
import csv
import torch
import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from data import get_stylegan_data, get_dff_data, get_eval_transform

RESULTS_FILE = "results.csv"

def extract_features(clip_model, data_loader, device):
    feats = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            # clip expects input normalized or in 0-1, preprocess was handled in transform
            f = clip_model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0).flatten()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running CLIP ViT-B-32 baseline on {device}...")
    
    # load pretrained CLIP
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    
    # probe training data: ~3000 samples from StyleGAN train split
    print("extracting CLIP features for 3000 training samples...")
    train_tf = get_eval_transform("clean")
    train_ds = get_stylegan_data(split="train", max_per_class=1500, transform=train_tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=False)
    
    X_train, y_train = extract_features(clip_model, train_loader, device)
    
    print("fitting logistic regression probe on frozen features...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    
    # evaluation sets and perturbations
    eval_sets = [
        ("StyleGAN", lambda tf: get_stylegan_data("test", max_per_class=1000, transform=tf)),
        ("SD-Inpaint", lambda tf: get_dff_data("inpainting", max_per_class=1000, transform=tf)),
        ("InsightFace", lambda tf: get_dff_data("insight", max_per_class=1000, transform=tf)),
        ("SD-Text2Img", lambda tf: get_dff_data("text2img", max_per_class=1000, transform=tf)),
    ]
    perturbations = ["clean", "jpeg90", "jpeg60", "resize0.5", "blur"]
    
    rows = []
    for set_name, ds_fn in eval_sets:
        for pert in perturbations:
            tf = get_eval_transform(pert)
            ds = ds_fn(tf)
            loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
            
            X_val, y_val = extract_features(clip_model, loader, device)
            preds_prob = clf.predict_proba(X_val)[:, 1]
            preds_bin = (preds_prob >= 0.5).astype(int)
            
            auc = roc_auc_score(y_val, preds_prob)
            acc = accuracy_score(y_val, preds_bin)
            print(f"clip_probe | {set_name:12s} | {pert:10s} | AUC: {auc:.4f} | Acc: {acc:.4f}")
            rows.append(["clip_probe", set_name, pert, round(auc, 6), round(acc, 4)])
            
    # append rows to results.csv
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "set", "perturbation", "auc", "acc"])
        writer.writerows(rows)
        
    print(f"appended {len(rows)} rows to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
