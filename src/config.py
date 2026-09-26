from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, CKPT, RESULTS = ROOT/"data", ROOT/"checkpoints", ROOT/"results"

SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 64
N_EVAL = 1000
NUM_WORKERS = 2
SG = DATA / "raw/stylegan/real_vs_fake/real-vs-fake"
DFF = DATA / "dff"    
TRAIN_DIRS = (SG / "train/real", SG / "train/fake")
VAL_DIRS = (SG / "valid/real", SG / "valid/fake")


SEEN = "StyleGAN"  #models trained on this
EVAL_SETS = {
    SEEN: (SG / "test/real", SG / "test/fake"),
    "SD-Inpaint": (DFF / "wiki", DFF / "inpainting"),
    "InsightFace": (DFF / "wiki", DFF / "insight"),
    "SD-Text2Img": (DFF / "wiki", DFF / "text2img"),
}