import os
import random
import re
import unicodedata
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "UBC-NLP/AraT5-base"
TRAIN_PATH = "train.csv"
VAL_PATH = "val.csv"
TEST_PATH = "test.csv"
OUTPUT_DIR = "./arat5_text_to_gloss_model"

MAX_SOURCE_LENGTH = 96
MAX_TARGET_LENGTH = 64
SEED = 42
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 12
WEIGHT_DECAY = 0.01

PREFIX = "حوّل النص الطبي إلى gloss: "


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_arabic_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }

    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("ـ", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_gloss(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(path):
    df = pd.read_csv(path)

    expected_cols = {"input_text", "target_gloss"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"{path} must contain these columns exactly: {expected_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["input_text"] = df["input_text"].astype(str).apply(normalize_arabic_text)
    df["target_gloss"] = df["target_gloss"].astype(str).apply(clean_gloss)

    df = df[
        (df["input_text"].str.strip() != "")
        & (df["target_gloss"].str.strip() != "")
    ].drop_duplicates(subset=["input_text", "target_gloss"]).reset_index(drop=True)

    return df


def build_dataset(train_df, val_df, test_df):
    return DatasetDict({
        "train": Dataset.from_pandas(train_df[["input_text", "target_gloss"]], preserve_index=False),
        "validation": Dataset.from_pandas(val_df[["input_text", "target_gloss"]], preserve_index=False),
        "test": Dataset.from_pandas(test_df[["input_text", "target_gloss"]], preserve_index=False),
    })


def exact_match_metric(preds, refs):
    total = len(preds)
    correct = sum(int(p.strip() == r.strip()) for p, r in zip(preds, refs))
    return correct / total if total > 0 else 0.0


def token_accuracy_metric(preds, refs):
    total_tokens = 0
    correct_tokens = 0

    for p, r in zip(preds, refs):
        p_tokens = p.strip().split()
        r_tokens = r.strip().split()

        max_len = max(len(p_tokens), len(r_tokens))
        for i in range(max_len):
            total_tokens += 1
            pred_tok = p_tokens[i] if i < len(p_tokens) else None
            ref_tok = r_tokens[i] if i < len(r_tokens) else None
            if pred_tok == ref_tok:
                correct_tokens += 1

    return correct_tokens / total_tokens if total_tokens > 0 else 0.0


def main():
    set_seed(SEED)

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    test_df = load_split(TEST_PATH)

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print("Unique train glosses:", train_df["target_gloss"].nunique())

    dataset = build_dataset(train_df, val_df, test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        inputs = [PREFIX + x for x in batch["input_text"]]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            text_target=batch["target_gloss"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        processing_class=tokenizer,
        model=model,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [clean_gloss(x) for x in decoded_preds]
        decoded_labels = [clean_gloss(x) for x in decoded_labels]

        em = exact_match_metric(decoded_preds, decoded_labels)
        tok_acc = token_accuracy_metric(decoded_preds, decoded_labels)

        return {
            "exact_match": em,
            "token_accuracy": tok_acc,
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...\n")
    trainer.train()

    print("\nEvaluating on test set...\n")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print("Test metrics:", test_metrics)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nGenerating predictions on test set...\n")
    preds = trainer.predict(tokenized["test"])
    predictions = preds.predictions

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    label_ids = preds.label_ids
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    decoded_preds = [clean_gloss(x) for x in decoded_preds]
    decoded_labels = [clean_gloss(x) for x in decoded_labels]

    results_df = test_df.copy()
    results_df["predicted_gloss"] = decoded_preds
    results_df["exact_match"] = (
        results_df["target_gloss"].str.strip() == results_df["predicted_gloss"].str.strip()
    ).astype(int)

    results_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved model to: {OUTPUT_DIR}")
    print(f"Saved test predictions to: {results_path}")

    print("\nExample inference:\n")
    examples = [
        "حبه 3 مرات قبل الاكل بنص ساعه",
        "لا توقف الدواء فجأه",
        "حط نقطتين بالعين اليمنى كل 8 ساعات",
    ]

    model.eval()
    device = model.device

    example_inputs = [PREFIX + normalize_arabic_text(x) for x in examples]
    enc = tokenizer(
        example_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
        )

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

    for src, out in zip(examples, decoded):
        print("INPUT :", src)
        print("GLOSS :", out)
        print("-" * 50)


if __name__ == "__main__":
    main()