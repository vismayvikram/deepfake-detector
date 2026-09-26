import random
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from src.config import DATA, N_EVAL, SEED
EXTS = (".jpg", ".jpeg", ".png")


if __name__ == "__main__":
    for name in ["wiki", "inpainting", "insight", "text2img"]:
        n = N_EVAL
        zip_path = hf_hub_download("OpenRL/DeepFakeFace", f"{name}.zip", repo_type="dataset")
        out = DATA / "dff" / name
        out.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as z:
            files = sorted(f for f in z.namelist() if f.lower().endswith(EXTS) and "__MACOSX" not in f)
            random.Random(SEED).shuffle(files)
            for i, f in enumerate(files[:n]):
                with z.open(f) as src, open(out / f"{i:05d}{Path(f).suffix.lower()}", "wb") as dst:
                    shutil.copyfileobj(src, dst)
        
        print(f"{name}: {len(list(out.iterdir()))} images -> {out}")