# data.py
# dataset loaders, download helpers, and augmentation transforms

import os
import io
import glob
import random
import zipfile
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# note: real images in stylegan split are FFHQ (high quality aligned studio faces)
# but in dff split real images are IMDB-WIKI (wild, noisy, varied resolution)
# so if unseen-generator generalization tanks, domain shift on reals is partly to blame!

STYLEGAN_DIR = "data/raw/stylegan/real_vs_fake/real-vs-fake"
DFF_DIR = "data/dff"

def download_stylegan():
    # helper if dataset isn't downloaded yet
    if os.path.exists(STYLEGAN_DIR):
        print("stylegan dataset already exists, skipping download")
        return
    print("downloading stylegan dataset from kaggle...")
    os.system("kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw/stylegan --unzip")

def download_dff():
    # helper for huggingface dff splits
    if os.path.exists(DFF_DIR) and len(os.listdir(DFF_DIR)) >= 4:
        print("dff dataset already exists, skipping download")
        return
    print("downloading dff from huggingface...")
    from huggingface_hub import hf_hub_download
    os.makedirs(DFF_DIR, exist_ok=True)
    zips = ["wiki.zip", "inpainting.zip", "insight.zip", "text2img.zip"]
    for z in zips:
        f = hf_hub_download(repo_id="OpenRL/DeepFakeFace", filename=z, repo_type="dataset")
        with zipfile.ZipFile(f, 'r') as zip_ref:
            zip_ref.extractall(DFF_DIR)

# --- perturbation helpers for eval ---
def apply_jpeg(img, quality=80):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def apply_resize(img, scale=0.5):
    # downsample then upsample back to simulate low-res sharing
    w, h = img.size
    small = img.resize((max(16, int(w * scale)), max(16, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)

def apply_blur(img, radius=1.0):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

# --- training augmentations ---
class RandomJpegAug:
    def __init__(self, p=0.4, min_q=40, max_q=90):
        self.p = p
        self.min_q = min_q
        self.max_q = max_q
    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(self.min_q, self.max_q)
            return apply_jpeg(img, quality=q)
        return img

class RandomBlurAug:
    def __init__(self, p=0.3, radius=1.0):
        self.p = p
        self.radius = radius
    def __call__(self, img):
        if random.random() < self.p:
            return apply_blur(img, radius=self.radius)
        return img

def get_train_transform(use_aug=False):
    tfs = [T.Resize((256, 256))]
    if use_aug:
        # casual augmentations to make model robust to compression and blur
        tfs.append(T.RandomHorizontalFlip(p=0.5))
        tfs.append(RandomJpegAug(p=0.4, min_q=40, max_q=90))
        tfs.append(RandomBlurAug(p=0.3, radius=1.0))
    tfs.append(T.ToTensor())
    return T.Compose(tfs)

def get_eval_transform(perturbation="clean"):
    def transform_fn(img):
        # perturbation applied before resizing or after? doing it on original then resize
        if perturbation == "jpeg90":
            img = apply_jpeg(img, 90)
        elif perturbation == "jpeg60":
            img = apply_jpeg(img, 60)
        elif perturbation == "resize0.5":
            img = apply_resize(img, 0.5)
        elif perturbation == "blur" or perturbation == "blur1.0":
            img = apply_blur(img, 1.0)
        # clean just resizes and converts to tensor
        img = img.resize((256, 256), Image.BILINEAR)
        return T.ToTensor()(img)
    return transform_fn

# --- dataset classes ---
class FaceDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize((256, 256))
            img = T.ToTensor()(img)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return img, y

def get_stylegan_data(split="train", max_per_class=None, transform=None):
    # split is train, valid, or test
    real_paths = sorted(glob.glob(os.path.join(STYLEGAN_DIR, split, "real", "*.jpg")))
    fake_paths = sorted(glob.glob(os.path.join(STYLEGAN_DIR, split, "fake", "*.jpg")))
    
    if max_per_class:
        # random seed to keep subsampling reproducible
        rng = random.Random(42)
        rng.shuffle(real_paths)
        rng.shuffle(fake_paths)
        real_paths = real_paths[:max_per_class]
        fake_paths = fake_paths[:max_per_class]
        
    paths = real_paths + fake_paths
    labels = [0.0] * len(real_paths) + [1.0] * len(fake_paths)
    return FaceDataset(paths, labels, transform=transform)

def get_dff_data(gen_name, max_per_class=None, transform=None):
    # gen_name in ["inpainting", "insight", "text2img"]
    # wiki is used as the real set for all unseen evaluations
    real_paths = sorted(glob.glob(os.path.join(DFF_DIR, "wiki", "*.jpg")))
    fake_paths = sorted(glob.glob(os.path.join(DFF_DIR, gen_name, "*.jpg")))
    
    if max_per_class:
        real_paths = real_paths[:max_per_class]
        fake_paths = fake_paths[:max_per_class]
        
    paths = real_paths + fake_paths
    labels = [0.0] * len(real_paths) + [1.0] * len(fake_paths)
    return FaceDataset(paths, labels, transform=transform)
