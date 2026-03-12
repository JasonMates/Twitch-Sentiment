import asyncio
import logging
import os
import re
import signal
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from scipy import sparse

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.deployment.twitch_listener import SimpleTwitchChatListener
except ImportError:
    from twitch_listener import SimpleTwitchChatListener

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv(project_root / ".env")
BOT_TOKEN = os.getenv("TWITCH_BOT_TOKEN")
BOT_NICK = os.getenv("TWITCH_BOT_NICK", "StreamAnalysisBot")

listener = None
classifier = None
stats = {
    "total": 0,
    "sentiments": Counter(),
    "start_time": time.time(),
}

_SPECIAL_RE = re.compile(r"""[!@#$%^&*()_+\-=\[\]{};:'",.<>?/\\|`~]""")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_MENTION_RE = re.compile(r"@\w+")
_EMOTE_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$")
_EPS = 1e-9


def signal_handler(sig, frame):
    global listener
    if listener:
        asyncio.create_task(listener.stop())


def get_sentiment_color(sentiment):
    colors = {
        "Positive": "\033[92m",
        "Negative": "\033[91m",
        "Neutral": "\033[93m",
    }
    return colors.get(sentiment, "\033[0m")


def normalize_bot_token(token: str) -> str:
    value = (token or "").strip()
    if value and not value.lower().startswith("oauth:"):
        value = f"oauth:{value}"
    return value


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


def _parse_prior_csv(value: str):
    if not value:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        return None
    try:
        arr = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    except ValueError:
        return None
    if np.any(arr <= 0):
        return None
    total = float(arr.sum())
    if total <= 0:
        return None
    return arr / total


class LogisticRegressionClassifier:
    def __init__(
            self,
            model_path: Path,
            min_confidence: float = 0.55,
            min_margin: float = 0.10,
            use_prior_correction: bool = False,
            target_priors_csv: str = "",
            train_priors_csv: str = "",
    ):
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.tfidf = payload["tfidf"]
        self.char_tfidf = payload["char_tfidf"]
        self.scaler = payload["scaler"]
        self.num_feature_names = payload["num_feature_names"]
        self.emote_sentiments = payload.get("emote_sentiments", {})
        self.label_names = payload.get("label_names", {0: "Negative", 1: "Neutral", 2: "Positive"})

        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.use_prior_correction = bool(use_prior_correction)

        self.id_for_label = {v: k for k, v in self.label_names.items()}
        self.neutral_id = self.id_for_label.get("Neutral", 1)

        parsed_target = _parse_prior_csv(target_priors_csv)
        parsed_train = _parse_prior_csv(train_priors_csv)

        self.target_priors = None
        self.train_priors = None
        if parsed_target is not None and parsed_train is not None:
            # Prior order is Negative, Neutral, Positive.
            self.target_priors = {
                self.id_for_label.get("Negative", 0): float(parsed_target[0]),
                self.id_for_label.get("Neutral", 1): float(parsed_target[1]),
                self.id_for_label.get("Positive", 2): float(parsed_target[2]),
            }
            self.train_priors = {
                self.id_for_label.get("Negative", 0): float(parsed_train[0]),
                self.id_for_label.get("Neutral", 1): float(parsed_train[1]),
                self.id_for_label.get("Positive", 2): float(parsed_train[2]),
            }
        else:
            self.use_prior_correction = False

    def _numeric_row(self, text: str) -> np.ndarray:
        em = extract_emote_features(text, self.emote_sentiments)
        tx = extract_text_features(text)
        merged = {}
        merged.update(em)
        merged.update(tx)
        values = [float(merged.get(name, 0.0)) for name in self.num_feature_names]
        arr = np.asarray(values, dtype=np.float32).reshape(1, -1)
        return self.scaler.transform(arr)

    def predict(self, text: str):
        msg = str(text or "")
        Xw = self.tfidf.transform([msg])
        Xc = self.char_tfidf.transform([msg])
        Xn = sparse.csr_matrix(self._numeric_row(msg))
        X = sparse.hstack([Xw, Xc, Xn], format="csr")

        raw_proba = self.model.predict_proba(X)[0].astype(np.float64)
        classes = [int(c) for c in self.model.classes_]

        if self.use_prior_correction and self.target_priors and self.train_priors:
            logp = np.log(np.clip(raw_proba, _EPS, 1.0))
            adjusted = []
            for i, cls in enumerate(classes):
                target = max(self.target_priors.get(cls, _EPS), _EPS)
                train = max(self.train_priors.get(cls, _EPS), _EPS)
                adjusted.append(logp[i] + np.log(target) - np.log(train))
            adjusted = np.asarray(adjusted, dtype=np.float64)
            adjusted = adjusted - np.max(adjusted)
            expv = np.exp(adjusted)
            proba = expv / expv.sum()
        else:
            proba = raw_proba

        order = np.argsort(proba)[::-1]
        idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else idx
        top_prob = float(proba[idx])
        margin = float(top_prob - float(proba[second_idx]))

        pred = int(classes[idx])
        if top_prob < self.min_confidence or margin < self.min_margin:
            pred = int(self.neutral_id)

        label = self.label_names.get(pred, "Neutral")
        confidence = top_prob
        return label, confidence


async def handle_message(msg_context):
    global classifier, stats
    sentiment, confidence = classifier.predict(msg_context.text)
    stats["total"] += 1
    stats["sentiments"][sentiment] += 1

    dt = datetime.fromtimestamp(msg_context.timestamp)
    time_str = dt.strftime("%H:%M:%S")
    color = get_sentiment_color(sentiment)
    reset = "\033[0m"
    conf_pct = f"{confidence * 100:.0f}%"

    emote_display = ""
    if msg_context.emotes:
        emotes_str = ", ".join(msg_context.emotes[:3])
        if len(msg_context.emotes) > 3:
            emotes_str += f" +{len(msg_context.emotes) - 3} more"
        emote_display = f" [{emotes_str}]"

    print(
        f"{time_str} {color}{sentiment:13s}{reset} ({conf_pct:4s}) | "
        f"{msg_context.username:20s} | {msg_context.text[:70]}{emote_display}"
    )

    if stats["total"] % 50 == 0:
        print_stats()


def print_stats():
    print("\n" + "-" * 100)
    elapsed = time.time() - stats["start_time"]
    mins, secs = divmod(int(elapsed), 60)
    rate = stats["total"] / elapsed if elapsed > 0 else 0
    print(f"Stats: {stats['total']} messages | {mins}m {secs}s | {rate:.1f} msg/s")

    if stats["total"] > 0:
        total = sum(stats["sentiments"].values())
        for sentiment in ["Positive", "Negative", "Neutral"]:
            count = stats["sentiments"][sentiment]
            pct = (count / total * 100) if total > 0 else 0
            color = get_sentiment_color(sentiment)
            reset = "\033[0m"
            bar = "#" * int(pct / 2)
            print(f"  {color}{sentiment:13s}{reset}: {bar} {count:3d} ({pct:4.1f}%)")
    print("-" * 100 + "\n")


async def main():
    global listener, classifier

    print("\n" + "=" * 50)
    print("TWITCH CHAT SENTIMENT ANALYZER")
    print("=" * 50)
    print()

    while True:
        channel = input("Enter Twitch channel name: ").strip().lower()
        if channel:
            break
        print("Channel name cannot be empty. Please try again.")

    print()

    if not BOT_TOKEN:
        print("\nERROR: Missing TWITCH_BOT_TOKEN!")
        print("\nCreate a .env file with:")
        print("TWITCH_BOT_TOKEN=your_token")
        return

    try:
        print("[1/3] Loading sentiment classifier...")
        model_path = project_root / "src" / "deployment" / "data" / "lr_sentiment_model.joblib"
        if not model_path.exists():
            model_path = Path(__file__).parent / "data" / "lr_sentiment_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                "Model not found. Train first with: "
                "python advanced_lr_model.py --model_out data/lr_sentiment_model.joblib"
            )
        lr_min_conf = float(os.getenv("LR_MIN_CONFIDENCE", "0.55"))
        lr_min_margin = float(os.getenv("LR_MIN_MARGIN", "0.10"))
        lr_use_prior = os.getenv("LR_USE_PRIOR_CORRECTION", "0").strip().lower() in {"1", "true", "yes", "on"}
        lr_target_priors = os.getenv("LR_TARGET_PRIORS", "")
        lr_train_priors = os.getenv("LR_TRAIN_PRIORS", "")

        classifier = LogisticRegressionClassifier(
            model_path=model_path,
            min_confidence=lr_min_conf,
            min_margin=lr_min_margin,
            use_prior_correction=lr_use_prior,
            target_priors_csv=lr_target_priors,
            train_priors_csv=lr_train_priors,
        )
        print(f"   Loaded: {model_path}")
        print(f"   Neutral gate: max_prob<{lr_min_conf:.2f} or margin<{lr_min_margin:.2f}")
        if lr_use_prior and lr_target_priors and lr_train_priors:
            print("   Prior correction: enabled")
        else:
            print("   Prior correction: disabled")

        print("[2/3] Connecting to Twitch IRC...")
        listener = SimpleTwitchChatListener(
            channel=channel,
            bot_token=normalize_bot_token(BOT_TOKEN),
            nickname=BOT_NICK,
            on_message_callback=handle_message,
        )

        print("[3/3] Starting chat listener...\n")
        print("Connected! Analyzing messages...\n")

        signal.signal(signal.SIGINT, signal_handler)
        await listener.start()

    except asyncio.CancelledError:
        print("\n\n  " + "=" * 96)
        print("    Shutting down...")
        if stats["total"] > 0:
            print("\n   FINAL STATISTICS")
            print("  " + "=" * 96)
            print_stats()
        print("\n  Shutdown complete\n")

    except Exception as e:
        print(f"\n  Error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
