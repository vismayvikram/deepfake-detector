# train.py
# training loop for spatial, freq, and fusion detectors

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from model import Detector
from data import get_stylegan_data, get_train_transform, get_eval_transform

# training only on StyleGAN split
# these are for testing generalization, not training on (never touch DFF during training!)

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

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training {args.mode} (aug={args.aug}) on {device}...")
    
    os.makedirs(args.save_dir, exist_ok=True)
    out_name = f"{args.mode}_aug.pt" if args.aug else f"{args.mode}.pt"
    save_path = os.path.join(args.save_dir, out_name)
    
    # datasets
    train_tf = get_train_transform(use_aug=args.aug)
    val_tf = get_eval_transform(perturbation="clean")
    
    train_ds = get_stylegan_data(split="train", max_per_class=args.max_samples, transform=train_tf)
    val_ds = get_stylegan_data(split="valid", max_per_class=1000, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    model = Detector(mode=args.mode)
    
    if args.freeze_freq and args.mode == "fusion":
        # pretrain freq on its own first since it was too weak to compete otherwise
        freq_path = os.path.join(args.save_dir, "freq_aug.pt" if args.aug else "freq.pt")
        if os.path.exists(freq_path):
            print(f"loading pretrained freq weights from {freq_path}")
            freq_state = torch.load(freq_path, map_location="cpu")
            model.load_state_dict(freq_state, strict=False)
            for param in model.freq_net.parameters():
                param.requires_grad = False
            for param in model.freq_norm.parameters():
                param.requires_grad = False
            print("freq branch frozen!")
        else:
            print(f"warning: {freq_path} not found, training freq from scratch instead")
            
    model = model.to(device)
    
    # adamw + cosine schedule
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0.0
    
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"ep {ep}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            
            main_out, s_out, f_out = model(x)
            
            # loss = BCE on main_out + ~0.3x BCE on each side output, in fusion mode
            if args.mode == "fusion":
                loss = bce(main_out, y) + 0.3 * bce(s_out, y) + 0.3 * bce(f_out, y)
            else:
                loss = bce(main_out, y)
                
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        sched.step()
        avg_train_loss = total_loss / max(1, n_batches)
        
        # evaluate on stylegan validation split
        val_auc, val_acc = eval_epoch(model, val_loader, device)
        print(f"epoch {ep}: train_loss={avg_train_loss:.4f} | val_auc={val_auc:.4f} | val_acc={val_acc:.4f}")
        
        # save whichever epoch had the best val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"--> new best val auc! saved checkpoint to {save_path}")
            
    print(f"done training {args.mode}. best val auc: {best_val_auc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="fusion", choices=["spatial", "freq", "fusion"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aug", action="store_true", help="use jpeg and blur augmentations")
    parser.add_argument("--freeze_freq", action="store_true", help="freeze pretrained freq branch")
    parser.add_argument("--max_samples", type=int, default=4000, help="max samples per class to keep training reasonable")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    train(args)
