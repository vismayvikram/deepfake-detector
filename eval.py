# eval.py
# runs every checkpoint on 4 sets x 5 perturbations
# saves results.csv and plots summary charts

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from model import Detector
from data import get_stylegan_data, get_dff_data, get_eval_transform

CHECKPOINT_DIR = "checkpoints"
RESULTS_FILE = "results.csv"

MODELS = [
    ("spatial", "spatial", "spatial.pt"),
    ("spatial_aug", "spatial", "spatial_aug.pt"),
    ("freq", "freq", "freq.pt"),
    ("freq_aug", "freq", "freq_aug.pt"),
    ("fusion", "fusion", "fusion.pt"),
    ("fusion_aug", "fusion", "fusion_aug.pt"),
]

SETS = [
    ("StyleGAN", lambda tf: get_stylegan_data("test", max_per_class=1000, transform=tf)),
    ("SD-Inpaint", lambda tf: get_dff_data("inpainting", max_per_class=1000, transform=tf)),
    ("InsightFace", lambda tf: get_dff_data("insight", max_per_class=1000, transform=tf)),
    ("SD-Text2Img", lambda tf: get_dff_data("text2img", max_per_class=1000, transform=tf)),
]

PERTURBATIONS = ["clean", "jpeg90", "jpeg60", "resize0.5", "blur"]

def evaluate_model_on_loader(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            main_out, _, _ = model(x)
            prob = torch.sigmoid(main_out).cpu().numpy().flatten()
            all_preds.extend(prob)
            all_targets.extend(y.numpy().flatten())
            
    auc = roc_auc_score(all_targets, all_preds)
    # reporting both auc and accuracy — notice if auc is ~0.60 but acc is 0.50
    # that means ranking slightly survived but 0.5 decision threshold failed completely
    acc = accuracy_score(all_targets, [1 if p >= 0.5 else 0 for p in all_preds])
    return auc, acc

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"evaluating checkpoints on {device}...")
    
    rows = []
    
    for model_name, mode, ckpt_file in MODELS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"skipping {model_name}, {ckpt_path} not found")
            continue
            
        print(f"\n--- evaluating {model_name} ({ckpt_file}) ---")
        model = Detector(mode=mode)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        
        for set_name, ds_fn in SETS:
            for pert in PERTURBATIONS:
                tf = get_eval_transform(pert)
                ds = ds_fn(tf)
                loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
                
                auc, acc = evaluate_model_on_loader(model, loader, device)
                print(f"{model_name:12s} | {set_name:12s} | {pert:10s} | AUC: {auc:.4f} | Acc: {acc:.4f}")
                rows.append([model_name, set_name, pert, round(auc, 6), round(acc, 4)])
                
    if not rows:
        print("no checkpoints found to evaluate!")
        return
        
    # save results to results.csv
    # if results.csv already exists, preserve any clip_probe rows
    existing_clip_rows = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if len(r) >= 5 and r[0] == "clip_probe":
                    existing_clip_rows.append(r)
                    
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "set", "perturbation", "auc", "acc"])
        writer.writerows(rows)
        writer.writerows(existing_clip_rows)
        
    print(f"\nwrote {len(rows) + len(existing_clip_rows)} total rows to {RESULTS_FILE}")
    
    # generate plots
    make_plots(rows + existing_clip_rows)

def make_plots(rows):
    print("generating summary plots...")
    data = {}
    for r in rows:
        m, s, p, auc, acc = r[0], r[1], r[2], float(r[3]), float(r[4])
        data.setdefault(m, {}).setdefault(s, {})[p] = (auc, acc)
        
    # Plot 1: Clean AUC across generator sets
    fig, ax = plt.subplots(figsize=(10, 6))
    sets = ["StyleGAN", "SD-Inpaint", "InsightFace", "SD-Text2Img"]
    models_to_plot = [m for m in ["spatial", "spatial_aug", "freq", "freq_aug", "fusion", "fusion_aug", "clip_probe"] if m in data]
    
    x = np.arange(len(sets))
    width = 0.12
    for i, m in enumerate(models_to_plot):
        aucs = [data[m].get(s, {}).get("clean", (0, 0))[0] for s in sets]
        ax.bar(x + (i - len(models_to_plot)/2) * width + width/2, aucs, width, label=m)
        
    ax.set_ylabel("AUC")
    ax.set_title("Clean Image AUC Across Seen & Unseen Generators")
    ax.set_xticks(x)
    ax.set_xticklabels(sets)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random Guess (0.5)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("auc_generator.png", dpi=150)
    plt.close()
    print("saved auc_generator.png")
    
    # Plot 2: Robustness against perturbations (on StyleGAN)
    fig, ax = plt.subplots(figsize=(9, 5))
    perts = ["clean", "jpeg90", "jpeg60", "resize0.5", "blur"]
    for m in models_to_plot:
        if m in data and "StyleGAN" in data[m]:
            aucs = [data[m]["StyleGAN"].get(p, (0, 0))[0] for p in perts]
            ax.plot(perts, aucs, marker="o", label=m)
            
    ax.set_ylabel("StyleGAN Test AUC")
    ax.set_xlabel("Perturbation")
    ax.set_title("Robustness to Compression & Blur Degradations (StyleGAN)")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("robustness.png", dpi=150)
    plt.close()
    print("saved robustness.png")

if __name__ == "__main__":
    main()
