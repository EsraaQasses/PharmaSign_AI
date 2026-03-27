import pandas as pd
import time
from jiwer import wer, cer
from app import transcribe_batch, normalize_text_strong

test_df = pd.read_csv("splits/test/test_fixed.csv")
sample = test_df.sample(n=min(5, len(test_df)), random_state=42)

hyps = transcribe_batch(sample["audio"].tolist(), use_vad=True)

for i, (idx, row) in enumerate(sample.iterrows()):
    ref = normalize_text_strong(row.text)
    hyp = hyps[i]
    print("\n---", i+1, "---")
    print("AUDIO:", row.audio)
    print("REF:", ref)
    print("HYP:", hyp)


train_df = pd.read_csv("splits/train/train_fixed.csv")
val_df   = pd.read_csv("splits/val/val_fixed.csv")
test_df  = pd.read_csv("splits/test/test_fixed.csv")

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print("Total files:", len(all_df))

results = []
refs_all, hyps_all = [], []
start = time.time()

BATCH = 4

for i in range(0, len(all_df), BATCH):
    batch = all_df.iloc[i:i+BATCH]
    paths = batch["audio"].tolist()

    try:
        hyps = transcribe_batch(paths, use_vad=True)
    except Exception as e:
        print("Error in batch", i, ":", e)
        hyps = [""] * len(batch)

    for j, (idx, row) in enumerate(batch.iterrows()):
        ref = normalize_text_strong(row["text"])
        hyp = hyps[j]

        refs_all.append(ref)
        hyps_all.append(hyp)

        results.append({
            "id": row.get("id", idx),
            "audio": row["audio"],
            "ref": ref,
            "hyp": hyp,
            "wer": wer([ref], [hyp]),
            "cer": cer([ref], [hyp])
        })

    if (i+1) % 20 == 0:
        print(f"{i+1}/{len(all_df)} | avg WER so far: {wer(refs_all, hyps_all):.4f}")

print("\nFINAL AVG WER:", wer(refs_all, hyps_all))
print("FINAL AVG CER:", cer(refs_all, hyps_all))
print("Minutes:", (time.time()-start)/60)

out_csv = "all_files_largev3_vad_norm_strong_predictions.csv"
pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")