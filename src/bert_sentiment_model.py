import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

DEFAULT_DATA_CSV = "Twitch_Sentiment_Labels.csv"
RANDOM_SEED = 42

LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
_EMOTE_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_AT_MENTION_RE = re.compile(r"@\w+")
_LEADING_SPEAKER_RE = re.compile(r"^\s*[A-Za-z0-9_]{2,25}:\s+")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_emote_lexicon(path: str) -> dict:
    lex = {}
    if not path or not os.path.exists(path):
        return lex
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            token = parts[0].strip()
            if not token:
                continue
            try:
                score = float(parts[1])
            except ValueError:
                continue
            lex[token] = score
            low = token.lower()
            if low not in lex:
                lex[low] = score
    return lex


def emote_candidates(token: str) -> list:
    base = str(token).strip()
    if not base:
        return []
    stripped = _EMOTE_EDGE_RE.sub("", base)
    raw = [base, base.lower(), stripped, stripped.lower()]
    out = []
    seen = set()
    for item in raw:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def augment_with_emote_tags(text: str, emote_lexicon: dict) -> str:
    msg = str(text or "")
    pos = 0
    neg = 0
    neu = 0

    for tok in msg.split():
        matched = None
        for cand in emote_candidates(tok):
            if cand in emote_lexicon:
                matched = float(emote_lexicon[cand])
                break
        if matched is None:
            continue
        if matched > 0.05:
            pos += 1
        elif matched < -0.05:
            neg += 1
        else:
            neu += 1

    tags = []
    if (pos + neg + neu) > 0:
        tags.append("has_emote")
    tags.extend(["emote_pos"] * min(pos, 3))
    tags.extend(["emote_neg"] * min(neg, 3))
    tags.extend(["emote_neu"] * min(neu, 3))

    if not tags:
        return msg
    return f"{msg} {' '.join(tags)}"


def normalize_twitter_text(text: str) -> str:
    msg = str(text or "")
    msg = _URL_RE.sub("http", msg)
    msg = _AT_MENTION_RE.sub("@user", msg)
    msg = _LEADING_SPEAKER_RE.sub("@user ", msg)
    return msg


def group_apply_no_groups(grouped, fn):
    try:
        return grouped.apply(fn, include_groups=False)
    except TypeError:
        return grouped.apply(fn)


def gold_vote_orig_id(g: pd.DataFrame) -> pd.Series:
    scores = g.groupby("sentiment")["confidence"].sum()
    top = scores.max()
    top_labels = scores[scores == top].index.tolist()

    if len(top_labels) == 1:
        label = top_labels[0]
        tie = False
    else:
        tie = True
        means = g[g["sentiment"].isin(top_labels)].groupby("sentiment")["confidence"].mean()
        topm = means.max()
        top2 = means[means == topm].index.tolist()
        if len(top2) == 1:
            label = top2[0]
        else:
            gg = g[g["sentiment"].isin(top2)].sort_values("timestamp_parsed")
            label = gg.iloc[-1]["sentiment"]

    message = g.sort_values("timestamp_parsed").iloc[-1]["message"]
    mean_conf = float(g[g["sentiment"] == label]["confidence"].mean())
    n_raters = int(g["labeled_by"].nunique())
    return pd.Series(
        {"message": message, "sentiment": label, "conf": mean_conf, "n_raters": n_raters, "tie": tie}
    )


def gold_vote_message(g: pd.DataFrame) -> pd.Series:
    tmp = g.copy()
    tmp["_w"] = (g["conf"].astype(float) * g["n_raters"].astype(float)).values

    scores = tmp.groupby("sentiment")["_w"].sum()
    top = scores.max()
    top_labels = scores[scores == top].index.tolist()

    if len(top_labels) == 1:
        label = top_labels[0]
    else:
        rcounts = tmp[tmp["sentiment"].isin(top_labels)].groupby("sentiment")["n_raters"].sum()
        top2 = rcounts[rcounts == rcounts.max()].index.tolist()
        label = top2[0]

    sub = tmp[tmp["sentiment"] == label]
    out_conf = float(np.average(sub["conf"].values, weights=sub["n_raters"].values))
    out_raters = int(tmp["n_raters"].max())
    return pd.Series({"sentiment": label, "conf": out_conf, "n_raters": out_raters})


def prepare_gold_message_df(data_csv: str) -> pd.DataFrame:
    df_raw = pd.read_csv(data_csv)
    required_cols = {"message_id", "message", "sentiment", "confidence", "labeled_by", "timestamp"}
    missing = sorted(required_cols.difference(df_raw.columns))
    if missing:
        raise ValueError(f"Missing required columns in {data_csv}: {missing}")

    df_raw["message"] = df_raw["message"].fillna("").astype(str)
    df_raw["sentiment"] = df_raw["sentiment"].astype(str).str.strip()
    df_raw = df_raw[df_raw["sentiment"].isin(LABEL_MAP)].copy()

    df = df_raw.copy()
    df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.sort_values(["message_id", "labeled_by", "timestamp_parsed"])
    df = df.groupby(["message_id", "labeled_by"], as_index=False).tail(1)

    df["message_id"] = df["message_id"].astype(str)
    df["is_src"] = df["message_id"].str.startswith("SRC-")

    df_num = df[~df["is_src"]].copy()
    df_src = df[df["is_src"]].copy()

    df_num["orig_id"] = pd.to_numeric(df_num["message_id"], errors="coerce")
    df_num = df_num.dropna(subset=["orig_id"]).copy()
    df_num["orig_id"] = df_num["orig_id"].astype(int)

    msg_to_ids = df_num.groupby("message")["orig_id"].unique().to_dict()
    mapped = []
    for message in df_src["message"]:
        ids = msg_to_ids.get(message, [])
        mapped.append(int(ids[0]) if len(ids) == 1 else np.nan)

    df_src["orig_id"] = mapped
    df_src = df_src.dropna(subset=["orig_id"]).copy()
    df_src["orig_id"] = df_src["orig_id"].astype(int)

    df_norm = pd.concat([df_num, df_src], ignore_index=True)
    df_norm["confidence"] = pd.to_numeric(df_norm["confidence"], errors="coerce").fillna(0.0)

    df_gold = group_apply_no_groups(df_norm.groupby("orig_id"), gold_vote_orig_id).reset_index()
    df_gold = df_gold[~df_gold["tie"]].copy()

    df_msg = group_apply_no_groups(df_gold.groupby("message"), gold_vote_message).reset_index()
    df_msg["sent_id"] = df_msg["sentiment"].map(LABEL_MAP).astype(int)
    return df_msg


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = int(self.labels[idx])
        return enc


def evaluate_split(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    macro = f1_score(y_true, y_pred, average="macro")
    micro = f1_score(y_true, y_pred, average="micro")
    acc = accuracy_score(y_true, y_pred)
    return {"split": name, "macro_f1": macro, "micro_f1": micro, "acc": acc}


@dataclass
class Splits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    prior_test_df: pd.DataFrame
    balanced_test_df: pd.DataFrame


def build_splits(df_msg: pd.DataFrame, test_size: float, seed: int) -> Splits:
    train_pool_idx, prior_test_idx = train_test_split(
        df_msg.index,
        test_size=test_size,
        random_state=seed,
        stratify=df_msg["sent_id"],
    )
    df_train_pool = df_msg.loc[train_pool_idx].copy()
    df_prior_test = df_msg.loc[prior_test_idx].copy()

    tr_idx, val_idx = train_test_split(
        df_train_pool.index,
        test_size=0.20,
        random_state=seed,
        stratify=df_train_pool["sent_id"],
    )
    df_tr = df_train_pool.loc[tr_idx].copy()
    df_val = df_train_pool.loc[val_idx].copy()

    min_n = int(df_prior_test["sentiment"].value_counts().min())
    sampled = [
        g.sample(n=min_n, random_state=seed)
        for _, g in df_prior_test.groupby("sentiment")
    ]
    df_bal_test = pd.concat(sampled, ignore_index=True)

    return Splits(train_df=df_tr, val_df=df_val, prior_test_df=df_prior_test, balanced_test_df=df_bal_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA_CSV)
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train_batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--warmup_pct", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--output_dir", default="data/bert_sentiment_model")
    ap.add_argument("--emote_lexicon", default="twitch_emote_vader_lexicon.txt")
    ap.add_argument("--no_emote_tags", action="store_true", help="Disable emote-tag text augmentation")
    ap.add_argument(
        "--normalize_twitter",
        action="store_true",
        help="Normalize URLs/mentions to Cardiff Twitter-style placeholders",
    )
    ap.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Enable early stopping when >0 (patience in eval epochs)",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("BERT SENTIMENT CLASSIFIER")
    print("=" * 60)

    df_msg = prepare_gold_message_df(args.data)
    print(f"\nDataset (gold by message): {len(df_msg)}")
    print(df_msg["sentiment"].value_counts())

    splits = build_splits(df_msg, test_size=args.test_size, seed=args.seed)
    print(f"\nTrain: {len(splits.train_df)} | Val: {len(splits.val_df)} | Prior test: {len(splits.prior_test_df)}")

    if args.normalize_twitter:
        for df_split in [splits.train_df, splits.val_df, splits.prior_test_df, splits.balanced_test_df]:
            df_split["message"] = df_split["message"].map(normalize_twitter_text)
        print("Using twitter-style normalization (URLs->http, mentions->@user)")

    use_emote_tags = not args.no_emote_tags
    emote_lexicon = {}
    if use_emote_tags:
        emote_lexicon = load_emote_lexicon(args.emote_lexicon)
        if emote_lexicon:
            for df_split in [splits.train_df, splits.val_df, splits.prior_test_df, splits.balanced_test_df]:
                df_split["message"] = df_split["message"].map(lambda x: augment_with_emote_tags(x, emote_lexicon))
            print(f"Using emote text augmentation with {len(emote_lexicon)} lexicon entries")
        else:
            use_emote_tags = False
            print("Emote lexicon not found or empty; emote text augmentation disabled")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=LABEL_NAMES,
        label2id=LABEL_MAP,
    )

    train_ds = TextDataset(
        splits.train_df["message"].tolist(),
        splits.train_df["sent_id"].tolist(),
        tokenizer,
        args.max_length,
    )
    val_ds = TextDataset(
        splits.val_df["message"].tolist(),
        splits.val_df["sent_id"].tolist(),
        tokenizer,
        args.max_length,
    )
    prior_ds = TextDataset(
        splits.prior_test_df["message"].tolist(),
        splits.prior_test_df["sent_id"].tolist(),
        tokenizer,
        args.max_length,
    )
    bal_ds = TextDataset(
        splits.balanced_test_df["message"].tolist(),
        splits.balanced_test_df["sent_id"].tolist(),
        tokenizer,
        args.max_length,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0 and args.warmup_pct > 0:
        steps_per_epoch = int(math.ceil(len(train_ds) / max(1, args.train_batch_size)))
        warmup_steps = int(max(0, round(steps_per_epoch * args.epochs * float(args.warmup_pct))))
        print(f"Auto warmup_steps={warmup_steps} from warmup_pct={args.warmup_pct:.3f}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_score(labels, pred, average="macro"),
            "micro_f1": f1_score(labels, pred, average="micro"),
            "acc": accuracy_score(labels, pred),
        }

    callbacks = []
    if args.early_stopping_patience and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(args.early_stopping_patience)))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    def predict_dataset(ds):
        out = trainer.predict(ds)
        pred = np.argmax(out.predictions, axis=-1)
        return pred

    y_val = splits.val_df["sent_id"].values
    y_prior = splits.prior_test_df["sent_id"].values
    y_bal = splits.balanced_test_df["sent_id"].values

    pred_val = predict_dataset(val_ds)
    pred_prior = predict_dataset(prior_ds)
    pred_bal = predict_dataset(bal_ds)

    rows = [
        evaluate_split("validation", y_val, pred_val),
        evaluate_split("balanced_test", y_bal, pred_bal),
        evaluate_split("prior_test", y_prior, pred_prior),
    ]

    print("\n" + "=" * 60)
    print("SPLIT METRICS")
    print("=" * 60)
    print(f"{'model':>8} {'split':>13} {'macro_f1':>9} {'micro_f1':>9} {'acc':>7}")
    for row in rows:
        print(f"{'BERT':>8} {row['split']:>13} {row['macro_f1']:9.4f} {row['micro_f1']:9.4f} {row['acc']:7.4f}")

    print("\n" + "-" * 60)
    print("PRIOR TEST - CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_prior, pred_prior, target_names=[LABEL_NAMES[i] for i in range(3)]))

    print("-" * 60)
    print("PRIOR TEST - CONFUSION MATRIX")
    print("-" * 60)
    print(confusion_matrix(y_prior, pred_prior))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    meta = {
        "use_emote_tags": use_emote_tags,
        "emote_lexicon_path": args.emote_lexicon,
        "max_length": args.max_length,
        "normalize_twitter": bool(args.normalize_twitter),
        "model_name": args.model_name,
    }
    with open(os.path.join(args.output_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved fine-tuned model to: {args.output_dir}")


if __name__ == "__main__":
    main()
