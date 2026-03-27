import zipfile, glob
from pathlib import Path
import pandas as pd, os, re

# audios.zip -> audios/
if Path("audios.zip").exists():
    with zipfile.ZipFile("audios.zip", "r") as z:
        z.extractall("audios")

# train/val/test zip -> splits/
for name in ["train.zip", "val.zip", "test.zip"]:
    if Path(name).exists():
        with zipfile.ZipFile(name, "r") as z:
            z.extractall("splits")

print("Audio files:", len(glob.glob("audios/**/*.ogg", recursive=True)))
print("Split files sample:", glob.glob("splits/**/*.*", recursive=True)[:15])


def fix_paths(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    df["audio"] = df["audio"].str.replace(r"^audios/", "audios/audios/", regex=True)
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print("saved:", csv_out)


fix_paths("splits/train/train.csv", "splits/train/train_fixed.csv")
fix_paths("splits/val/val.csv", "splits/val/val_fixed.csv")
fix_paths("splits/test/test.csv", "splits/test/test_fixed.csv")

t = pd.read_csv("splits/train/train_fixed.csv")
print("example:", t.iloc[0]["audio"])
print("exists?", os.path.exists(t.iloc[0]["audio"]))