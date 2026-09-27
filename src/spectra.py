import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import EVAL_SETS, RESULTS
from src.data import get_transforms, list_images
from src.frequency import log_spectrum


def mean_spectrum(folder, n=1000):
    tf = get_transforms(train=False, aug=False)
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in list_images(folder, limit=n)])
    return log_spectrum(x, normalize=False).mean(0)[0]


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(EVAL_SETS), 3, figsize=(9, 3 * len(EVAL_SETS)), layout="tight")
    axes = np.atleast_2d(axes)
    for row, (name, (real, fake)) in zip(axes, EVAL_SETS.items()):
        real_spec = mean_spectrum(real)
        fake_spec = mean_spectrum(fake)
        diff_spec = fake_spec - real_spec
        lim = diff_spec.abs().max().item()
        data = [
            (real_spec, f"{name}: real", "magma", None, None),
            (fake_spec, f"{name}: fake", "magma", None, None),
            (diff_spec, f"{name}: fake - real", "coolwarm", -lim, lim),
        ]
        for ax, (img, title, cmap, vmin, vmax) in zip(row, data):
            ax.imshow(img.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    out_path = RESULTS / "mean_spectra.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")