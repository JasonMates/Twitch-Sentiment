import argparse
import os
import re
import warnings
import joblib

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# CONFIG defaults can be overridden via CLI, use the cli

DEFAULT_DATA_CSV = "Twitch_Sentiment_Labels.csv"
DEFAULT_EMOTE_LEXICON = "twitch_emote_vader_lexicon.txt"

RANDOM_SEED = 42

WORD_TFIDF = dict(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    lowercase=True,
)

CHAR_TFIDF = dict(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.98,
    max_features=8000,
    lowercase=True,
)

C_GRID = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]

LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
DEFAULT_CLASS_WEIGHT = None

_SPECIAL_RE = re.compile(r"""[!@#$%^&*()_+\-=\[\]{};:'",.<>?/\\|`~]""")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_MENTION_RE = re.compile(r"@\w+")
_EMOTE_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$")


# HELPERS

def group_apply_no_groups(grouped, fn):
    try:
        return grouped.apply(fn, include_groups=False)
    except TypeError:
        return grouped.apply(fn)


def make_logreg(C: float, class_weight, max_iter: int, n_jobs: int) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        penalty="l2",
        solver="saga",
        class_weight=class_weight,
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
    )


def load_emote_lexicon(path: str) -> dict:
    """
    VADER lexicon format:
      <TOKEN>\\t<SCORE>
    extra columns are ignored
    """
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

    # add lowercase aliases to increase match rate without changing existing keys
    for token, score in list(lex.items()):
        low = token.lower()
        if low not in lex:
            lex[low] = score
    return lex


def emote_candidates(token: str) -> list:
    base = token.strip()
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


def extract_emote_features(text: str, emote_sentiments: dict) -> dict:
    scores = []
    pos_count = 0
    neg_count = 0

    for token in str(text).split():
        matched = None
        for cand in emote_candidates(token):
            if cand in emote_sentiments:
                matched = emote_sentiments[cand]
                break
        if matched is None:
            continue
        scores.append(matched)
        if matched > 0:
            pos_count += 1
        elif matched < 0:
            neg_count += 1

    if scores:
        s = np.asarray(scores, dtype=np.float32)
        return {
            "emote_count": float(len(s)),
            "emote_avg_sentiment": float(np.mean(s)),
            "emote_max_sentiment": float(np.max(s)),
            "emote_min_sentiment": float(np.min(s)),
            "emote_sum_sentiment": float(np.sum(s)),
            "emote_positive_count": float(pos_count),
            "emote_negative_count": float(neg_count),
            "emote_abs_sum": float(np.sum(np.abs(s))),
            "has_positive_emote": float(pos_count > 0),
            "has_negative_emote": float(neg_count > 0),
            "emote_variance": float(np.var(s)),
        }

    return {
        "emote_count": 0.0,
        "emote_avg_sentiment": 0.0,
        "emote_max_sentiment": 0.0,
        "emote_min_sentiment": 0.0,
        "emote_sum_sentiment": 0.0,
        "emote_positive_count": 0.0,
        "emote_negative_count": 0.0,
        "emote_abs_sum": 0.0,
        "has_positive_emote": 0.0,
        "has_negative_emote": 0.0,
        "emote_variance": 0.0,
    }


def extract_text_features(text: str) -> dict:
    message = str(text)
    words = message.split()
    word_count = float(len(words))
    unique_ratio = (len(set(words)) / word_count) if word_count else 0.0

    return {
        "word_count": word_count,
        "char_count": float(len(message)),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "has_caps": float(any(c.isupper() for c in message)),
        "all_caps": float(message.isupper() and len(message) > 2),
        "exclamation_count": float(message.count("!")),
        "question_count": float(message.count("?")),
        "repeated_chars": float(bool(_REPEAT_RE.search(message))),
        "repeated_chars_count": float(len(_REPEAT_RE.findall(message))),
        "mention_count": float(len(_MENTION_RE.findall(message))),
        "special_chars": float(len(_SPECIAL_RE.findall(message))),
        "digit_count": float(sum(c.isdigit() for c in message)),
        "unique_word_ratio": float(unique_ratio),
    }


def build_numeric_features(messages: pd.Series, emote_sentiments: dict) -> tuple:
    em_df = pd.DataFrame([extract_emote_features(t, emote_sentiments) for t in messages], dtype=np.float32)
    tx_df = pd.DataFrame([extract_text_features(t) for t in messages], dtype=np.float32)
    num_feature_names = list(em_df.columns) + list(tx_df.columns)
    X_num = np.hstack([em_df.values, tx_df.values]).astype(np.float32)
    return X_num, num_feature_names


def build_feature_matrices(train_df: pd.DataFrame, eval_dfs: list, emote_sentiments: dict):
    tfidf = TfidfVectorizer(**WORD_TFIDF)
    char_tfidf = TfidfVectorizer(**CHAR_TFIDF)

    train_messages = train_df["message"].astype(str)

    Xw_train = tfidf.fit_transform(train_messages)
    Xc_train = char_tfidf.fit_transform(train_messages)
    Xnum_train, num_feature_names = build_numeric_features(train_messages, emote_sentiments)

    scaler = StandardScaler()
    Xnum_train = scaler.fit_transform(Xnum_train)

    X_train = sparse.hstack(
        [Xw_train, Xc_train, sparse.csr_matrix(Xnum_train)],
        format="csr",
    )

    eval_mats = []
    for df_eval in eval_dfs:
        eval_messages = df_eval["message"].astype(str)
        Xw_eval = tfidf.transform(eval_messages)
        Xc_eval = char_tfidf.transform(eval_messages)
        Xnum_eval, _ = build_numeric_features(eval_messages, emote_sentiments)
        Xnum_eval = scaler.transform(Xnum_eval)
        X_eval = sparse.hstack(
            [Xw_eval, Xc_eval, sparse.csr_matrix(Xnum_eval)],
            format="csr",
        )
        eval_mats.append(X_eval)

    bundle = {
        "tfidf": tfidf,
        "char_tfidf": char_tfidf,
        "scaler": scaler,
        "num_feature_names": num_feature_names,
    }
    return X_train, eval_mats, bundle


def compute_sample_weights(df: pd.DataFrame, scale_max: float = None) -> tuple:
    weights = (df["conf"].astype(float) * df["n_raters"].astype(float)).values.astype(np.float32)
    if scale_max is None:
        scale_max = max(float(weights.max()), 1.0)
    return (weights / scale_max).astype(np.float32), float(scale_max)


def gold_vote_orig_id(g: pd.DataFrame) -> pd.Series:
    """
    gold label per orig_id using confidence-weighted vote.
    tie policy:
    1) highest total confidence wins
    2) i tie: highest mean confidence wins
    3) if still tie: most recent label wins
    """
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


def leakage_check(train_texts: pd.Series, test_texts: pd.Series, label: str) -> None:
    overlap = set(train_texts).intersection(set(test_texts))
    print(f"Leakage check ({label}): {len(overlap)} shared messages between train and test.")


def evaluate_split(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    macro = f1_score(y_true, y_pred, average="macro")
    micro = f1_score(y_true, y_pred, average="micro")
    acc = accuracy_score(y_true, y_pred)
    return {"split": name, "macro_f1": macro, "micro_f1": micro, "acc": acc}


def print_confusion(cm: np.ndarray) -> None:
    print(f"\n{'':10}Negative  Neutral  Positive")
    for i, lab in enumerate(["Negative", "Neutral", "Positive"]):
        print(f"{lab:10}{cm[i, 0]:8d}{cm[i, 1]:8d}{cm[i, 2]:8d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=DEFAULT_DATA_CSV,
        help="CSV with columns: message_id,message,sentiment,confidence,labeled_by,timestamp",
    )
    ap.add_argument(
        "--lexicon",
        default=DEFAULT_EMOTE_LEXICON,
        help="VADER-style emote lexicon (token\\tscore)",
    )
    ap.add_argument("--test_size", type=float, default=0.25, help="Holdout test size on gold-by-message set")
    ap.add_argument("--balanced", action="store_true", help="Use class_weight='balanced'")
    ap.add_argument("--show_thresholded", action="store_true", help="Also print thresholded results")
    ap.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel workers for LogisticRegression (-1 uses all cores)",
    )
    ap.add_argument("--max_iter", type=int, default=5000, help="Max iterations for LogisticRegression")
    ap.add_argument(
        "--model_out",
        default="data/lr_sentiment_model.joblib",
        help="Output path for serialized model bundle",
    )
    args = ap.parse_args()

    data_csv = args.data
    class_weight = "balanced" if args.balanced else DEFAULT_CLASS_WEIGHT

    df_raw = pd.read_csv(data_csv)
    required_cols = {"message_id", "message", "sentiment", "confidence", "labeled_by", "timestamp"}
    missing = sorted(required_cols.difference(df_raw.columns))
    if missing:
        raise ValueError(f"Missing required columns in {data_csv}: {missing}")

    # keep only known labels and normalize message text
    df_raw["message"] = df_raw["message"].fillna("").astype(str)
    df_raw["sentiment"] = df_raw["sentiment"].astype(str).str.strip()
    df_raw = df_raw[df_raw["sentiment"].isin(LABEL_MAP)].copy()

    print("=" * 60)
    print("ADVANCED LOGISTIC REGRESSION SENTIMENT CLASSIFIER")
    print("=" * 60)
    print(f"\nDataset (raw rows): {len(df_raw)} samples")
    print(f"Sentiment distribution (raw rows):\n{df_raw['sentiment'].value_counts()}\n")

    # NORMALIZE LABELS
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

    print(f"Dataset (gold orig_id): {len(df_gold)} unique orig_id")
    print("Sentiment distribution (gold orig_id):")
    print(df_gold["sentiment"].value_counts())
    print("Rater coverage (gold orig_id):")
    print(df_gold["n_raters"].value_counts())

    df_msg = group_apply_no_groups(df_gold.groupby("message"), gold_vote_message).reset_index()

    print(f"\nDataset (gold by message): {len(df_msg)} unique message strings")
    print("Sentiment distribution (gold by message):")
    print(df_msg["sentiment"].value_counts())
    print("Rater coverage proxy (gold by message, max per message):")
    print(df_msg["n_raters"].value_counts())

    # SPLITS
    df_use = df_msg.copy()
    df_use["sent_id"] = df_use["sentiment"].map(LABEL_MAP).astype(int)

    train_pool_idx, prior_test_idx = train_test_split(
        df_use.index,
        test_size=args.test_size,
        random_state=RANDOM_SEED,
        stratify=df_use["sent_id"],
    )

    df_train_pool = df_use.loc[train_pool_idx].copy()
    df_prior_test = df_use.loc[prior_test_idx].copy()

    tr_idx, val_idx = train_test_split(
        df_train_pool.index,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df_train_pool["sent_id"],
    )
    df_tr = df_train_pool.loc[tr_idx].copy()
    df_val = df_train_pool.loc[val_idx].copy()

    leakage_check(df_tr["message"], df_prior_test["message"], "by message (train vs prior_test)")
    leakage_check(df_tr["message"], df_val["message"], "by message (train vs validation)")

    min_n = int(df_prior_test["sentiment"].value_counts().min())
    df_bal_test = (
        df_prior_test.groupby("sentiment", group_keys=False)
        .apply(lambda g: g.sample(n=min_n, random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )

    # LOAD LEXICON
    emote_sentiments = load_emote_lexicon(args.lexicon)
    print(f"Loaded emote lexicon entries: {len(emote_sentiments)}")


    # HYPERPARAMETER TUNING
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING (5-fold Stratified CV)")
    print("=" * 60)

    y_tr = df_tr["sent_id"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    best_score = -1.0
    best_C = C_GRID[0]

    for C in C_GRID:
        fold_scores = []
        for fold_tr_i, fold_cv_i in skf.split(df_tr["message"], y_tr):
            fold_tr = df_tr.iloc[fold_tr_i].copy()
            fold_cv = df_tr.iloc[fold_cv_i].copy()

            X_fold_tr, [X_fold_cv], _ = build_feature_matrices(fold_tr, [fold_cv], emote_sentiments)
            y_fold_tr = fold_tr["sent_id"].values
            y_fold_cv = fold_cv["sent_id"].values
            w_fold_tr, _ = compute_sample_weights(fold_tr)

            lr = make_logreg(
                C=C,
                class_weight=class_weight,
                max_iter=args.max_iter,
                n_jobs=args.n_jobs,
            )
            lr.fit(X_fold_tr, y_fold_tr, sample_weight=w_fold_tr)
            pred_cv = lr.predict(X_fold_cv)
            fold_scores.append(f1_score(y_fold_cv, pred_cv, average="macro"))

        cv_mean = float(np.mean(fold_scores))
        print(f"  Penalty=l2, C={C:5.2f} -> CV F1 (macro): {cv_mean:.4f}")

        if cv_mean > best_score:
            best_score = cv_mean
            best_C = C

    print(f"\n{'BEST HYPERPARAMETERS':-^60}")
    print(f"C = {best_C}, Penalty = l2")
    print(f"CV F1 (macro) = {best_score:.4f}")

    # VALIDATION MODEL
    X_train_val, [X_val_eval], _ = build_feature_matrices(df_tr, [df_val], emote_sentiments)
    y_train_val = df_tr["sent_id"].values
    y_val = df_val["sent_id"].values
    w_train_val, _ = compute_sample_weights(df_tr)

    val_model = make_logreg(
        C=best_C,
        class_weight=class_weight,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
    )
    val_model.fit(X_train_val, y_train_val, sample_weight=w_train_val)
    pred_val = val_model.predict(X_val_eval)

    # FINAL MODEL
    print(f"\n{'FINAL MODEL TRAINING':-^60}")

    X_train_full, [X_prior, X_bal], feature_bundle = build_feature_matrices(
        df_train_pool,
        [df_prior_test, df_bal_test],
        emote_sentiments,
    )

    y_train_full = df_train_pool["sent_id"].values
    y_prior = df_prior_test["sent_id"].values
    y_bal = df_bal_test["sent_id"].values
    w_train_full, _ = compute_sample_weights(df_train_pool)

    model = make_logreg(
        C=best_C,
        class_weight=class_weight,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
    )
    model.fit(X_train_full, y_train_full, sample_weight=w_train_full)

    tfidf_count = len(feature_bundle["tfidf"].get_feature_names_out())
    char_count = len(feature_bundle["char_tfidf"].get_feature_names_out())
    num_count = len(feature_bundle["num_feature_names"])
    print(
        f"Total features: {X_train_full.shape[1]} "
        f"(TF-IDF: {tfidf_count}, Char: {char_count}, Numeric: {num_count})"
    )

    if args.show_thresholded:
        probs = model.predict_proba(X_prior)
        t_neg = 0.30
        t_pos = 0.45

        y_thr = np.full_like(y_prior, fill_value=1)
        y_thr[probs[:, 0] >= t_neg] = 0
        y_thr[(y_thr == 1) & (probs[:, 2] >= t_pos)] = 2

        print("\n" + "-" * 60)
        print("THRESHOLDED RESULTS (prior_test)")
        print("-" * 60)
        print("Macro F1:", f1_score(y_prior, y_thr, average="macro"))
        print("Micro F1:", f1_score(y_prior, y_thr, average="micro"))
        print("Accuracy:", accuracy_score(y_prior, y_thr))
        print(classification_report(y_prior, y_thr, target_names=[LABEL_NAMES[i] for i in range(3)]))
        print(confusion_matrix(y_prior, y_thr))

    # EVAL
    pred_bal = model.predict(X_bal)
    pred_prior = model.predict(X_prior)

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
        print(f"{'LogReg':>8} {row['split']:>13} {row['macro_f1']:9.4f} {row['micro_f1']:9.4f} {row['acc']:7.4f}")

    print("\n" + "-" * 60)
    print("PRIOR TEST - CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_prior, pred_prior, target_names=[LABEL_NAMES[i] for i in range(3)]))

    print(f"{'PRIOR TEST - CONFUSION MATRIX':-^60}")
    cm = confusion_matrix(y_prior, pred_prior)
    print_confusion(cm)

    # TOP FEATURES
    print(f"\n{'TOP FEATURES':-^60}")

    tfidf_names = [f"tfidf:{token}" for token in feature_bundle["tfidf"].get_feature_names_out()]
    char_names = [f"char:{token}" for token in feature_bundle["char_tfidf"].get_feature_names_out()]
    num_names = feature_bundle["num_feature_names"]
    feature_names = tfidf_names + char_names + num_names

    avg_coef = np.abs(model.coef_).mean(axis=0)
    if len(feature_names) != len(avg_coef):
        raise ValueError(
            f"Feature name count mismatch: names={len(feature_names)} vs coefs={len(avg_coef)}. "
            f"TFIDF={len(tfidf_names)}, CHAR={len(char_names)}, NUM={len(num_names)}"
        )

    top_idx = np.argsort(avg_coef)[-20:][::-1]
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank:2d}. {feature_names[idx]:30s} {avg_coef[idx]:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    model_payload = {
        "model": model,
        "tfidf": feature_bundle["tfidf"],
        "char_tfidf": feature_bundle["char_tfidf"],
        "scaler": feature_bundle["scaler"],
        "num_feature_names": feature_bundle["num_feature_names"],
        "emote_sentiments": emote_sentiments,
        "label_names": LABEL_NAMES,
        "label_map": LABEL_MAP,
        "best_c": best_C,
    }

    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model_payload, args.model_out)
    print(f"Saved model artifact to: {args.model_out}")


if __name__ == "__main__":
    main()
