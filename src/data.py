import io
import random
from pathlib import Path
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from src.config import BATCH_SIZE, IMG_SIZE, NUM_WORKERS, SEED
EXTS = {".jpg", ".jpeg", ".png"}

def list_images(folder, limit=None):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)
    if limit:
        random.Random(SEED).shuffle(files)   # fixed seed -> same subset every run
        files = files[:limit]
    return files

def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

class RandomJPEG:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self,img):
        if random.randint() > self.p:
            return jpeg(img, random.randint(50,100)) 
        return img

class RandomBlur:
    def __init(self, p=0.2):
        self.p = p
    def __call__(self,img):
        if random.randint()> self.p:
            return img.filter(ImageFilter.GaussianBlue(radius=random.uniform(0.1, 1.5)))
        return img


def apply_perturbations(img, kind, level):
    if kind == "jpeg":
        return jpeg(img, level)
    elif kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=level))
    elif kind == "resize":
        small_size = int(IMG_SIZE * level)
        img = TF.resize(img, [small_size, small_size])
        return TF.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

def get_transforms(train=False, aug=False, perturb=None):
    ops = [T.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        ops.append(T.RandomHorizontalFlip())
        if aug:
            ops.extend([RandomJPEG(), RandomBlur()])
    if perturb is not None:
        kind, level = perturb
        ops.append(T.Lambda(lambda img: apply_perturbations(img, kind, level)))

    ops.append(T.ToTensor())
    return T.Compose(ops)

class RealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, limit=None, transform=None):
        real_paths = list_images(real_dir, limit)
        fake_paths = list_images(fake_dir, limit)

        # Store simple (path, integer_label) pairs directly
        self.samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)
    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

def make_loader(dataset, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )





