# train.py
# Single-run full training pipeline for spatial, freq, and fusion detectors (clean & augmented, without tqdm)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from model import Detector
from data import get_stylegan_data, get_train_transform, get_eval_transform


def eval_epoch(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            main_out, _, _ = model(x)
            prob = torch.sigmoid(main_out).cpu().numpy().flatten()
            all_preds.extend(prob)
            all_targets.extend(y.cpu().numpy().flatten())
            
    auc = roc_auc_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, [1 if p >= 0.5 else 0 for p in all_preds])
    return auc, acc


def train_model(mode, epochs, batch_size, lr, aug, freeze_freq, max_samples, save_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Training Mode: {mode.upper()} | Augmentation: {aug} | Device: {device}")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    out_name = f"{mode}_aug.pt" if aug else f"{mode}.pt"
    save_path = os.path.join(save_dir, out_name)
    
    # Datasets and transforms
    train_tf = get_train_transform(use_aug=aug)
    val_tf = get_eval_transform(perturbation="clean")
    
    train_ds = get_stylegan_data(split="train", max_per_class=max_samples, transform=train_tf)
    val_ds = get_stylegan_data(split="valid", max_per_class=500, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = Detector(mode=mode)
    
    # Handle frequency branch pretraining/freezing for fusion mode
    if freeze_freq and mode == "fusion":
        freq_path = os.path.join(save_dir, "freq_aug.pt" if aug else "freq.pt")
        if os.path.exists(freq_path):
            print(f"Loading pretrained frequency weights from {freq_path}")
            freq_state = torch.load(freq_path, map_location="cpu")
            model.load_state_dict(freq_state, strict=False)
            for param in model.freq_net.parameters():
                param.requires_grad = False
            for param in model.freq_norm.parameters():
                param.requires_grad = False
            print("Frequency branch frozen successfully!")
        else:
            print(f"Warning: {freq_path} not found, training frequency branch from scratch instead.")
            
    model = model.to(device)
    
    # Optimizer and scheduler setup
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0.0
    
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            
            main_out, s_out, f_out = model(x)
            
            if mode == "fusion":
                loss = bce(main_out, y) + 0.3 * bce(s_out, y) + 0.3 * bce(f_out, y)
            else:
                loss = bce(main_out, y)
                
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        sched.step()
        avg_train_loss = total_loss / max(1, n_batches)
        
        # Evaluate on validation split
        val_auc, val_acc = eval_epoch(model, val_loader, device)
        print(f"Epoch {ep}/{epochs} -> Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"--> New best val AUC! Saved checkpoint to {save_path}")
            
    print(f"Finished training {mode} (aug={aug}). Best Val AUC: {best_val_auc:.4f}\n")


def main():
    # Pipeline hyperparameters
    epochs = 3
    batch_size = 32
    lr = 1e-4
    max_samples = 4000
    save_dir = "checkpoints"

    # Full execution matrix: runs clean models first, then augmented models, 
    # ensuring frequency models are trained before fusion models load them.
    pipeline_configs = [
        {"mode": "spatial", "aug": False, "freeze_freq": False},
        {"mode": "freq",    "aug": False, "freeze_freq": False},
        {"mode": "fusion",  "aug": False, "freeze_freq": True},
        {"mode": "spatial", "aug": True,  "freeze_freq": False},
        {"mode": "freq",    "aug": True,  "freeze_freq": False},
        {"mode": "fusion",  "aug": True,  "freeze_freq": True},
    ]

    print("Starting full end-to-end training pipeline...")
    for config in pipeline_configs:
        train_model(
            mode=config["mode"],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            aug=config["aug"],
            freeze_freq=config["freeze_freq"],
            max_samples=max_samples,
            save_dir=save_dir
        )
    print("All training runs completed successfully!")


if __name__ == "__main__":
    main()