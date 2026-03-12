import asyncio
import json
import logging
import os
import re
import signal
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from src.deployment.twitch_listener import SimpleTwitchChatListener
except ImportError:
    from twitch_listener import SimpleTwitchChatListener

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent


def _load_env_from_nearest_parent() -> None:
    for base in [APP_DIR, *APP_DIR.parents]:
        env_path = base / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return
    load_dotenv()


_load_env_from_nearest_parent()
BOT_TOKEN = os.getenv("TWITCH_BOT_TOKEN")
BOT_NICK = os.getenv("TWITCH_BOT_NICK", "StreamAnalysisBot")

listener = None
classifier = None
stats = {
    "total": 0,
    "sentiments": Counter(),
    "start_time": time.time(),
}
_EMOTE_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_AT_MENTION_RE = re.compile(r"@\w+")
_LEADING_SPEAKER_RE = re.compile(r"^\s*[A-Za-z0-9_]{2,25}:\s+")
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


class BertSentimentClassifier:
    def __init__(
            self,
            model_dir: Path,
            max_length: int = 128,
            use_emote_tags: bool = False,
            emote_lexicon_path: str = "",
            min_confidence: float = 0.55,
            min_margin: float = 0.10,
            use_prior_correction: bool = False,
            target_priors_csv: str = "",
            train_priors_csv: str = "",
            normalize_twitter: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_emote_tags = bool(use_emote_tags)
        self.normalize_twitter = bool(normalize_twitter)
        self.emote_sentiments = self._load_emote_lexicon(emote_lexicon_path) if self.use_emote_tags else {}
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.use_prior_correction = bool(use_prior_correction)

        id2label = getattr(self.model.config, "id2label", None) or {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
        }
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.neutral_id = int(self.label2id.get("Neutral", 1))

        self.target_priors = self._parse_prior_csv(target_priors_csv)
        self.train_priors = self._parse_prior_csv(train_priors_csv)
        if not (self.target_priors and self.train_priors):
            self.use_prior_correction = False

    @staticmethod
    def _load_emote_lexicon(path: str) -> dict:
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

    def _parse_prior_csv(self, csv_value: str):
        if not csv_value:
            return None
        parts = [p.strip() for p in csv_value.split(",")]
        if len(parts) != 3:
            return None
        try:
            vals = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
        except ValueError:
            return None
        if np.any(vals <= 0):
            return None
        vals = vals / vals.sum()
        # Expected order: Negative, Neutral, Positive
        return {
            int(self.label2id.get("Negative", 0)): float(vals[0]),
            int(self.label2id.get("Neutral", 1)): float(vals[1]),
            int(self.label2id.get("Positive", 2)): float(vals[2]),
        }

    @staticmethod
    def _emote_candidates(token: str) -> list:
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

    def _augment_with_emote_tags(self, text: str) -> str:
        msg = str(text or "")
        if not self.emote_sentiments:
            return msg

        pos = 0
        neg = 0
        neu = 0
        for tok in msg.split():
            matched = None
            for cand in self._emote_candidates(tok):
                if cand in self.emote_sentiments:
                    matched = float(self.emote_sentiments[cand])
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

    @staticmethod
    def _normalize_twitter_text(text: str) -> str:
        msg = str(text or "")
        msg = _URL_RE.sub("http", msg)
        msg = _AT_MENTION_RE.sub("@user", msg)
        msg = _LEADING_SPEAKER_RE.sub("@user ", msg)
        return msg

    @torch.no_grad()
    def predict(self, text: str):
        msg = str(text or "")
        if self.normalize_twitter:
            msg = self._normalize_twitter_text(msg)
        if self.use_emote_tags:
            msg = self._augment_with_emote_tags(msg)
        enc = self.tokenizer(
            msg,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float64)

        class_ids = [int(i) for i in range(len(probs))]
        if self.use_prior_correction and self.target_priors and self.train_priors:
            logp = np.log(np.clip(probs, _EPS, 1.0))
            adjusted = []
            for i, cls in enumerate(class_ids):
                target = max(self.target_priors.get(cls, _EPS), _EPS)
                train = max(self.train_priors.get(cls, _EPS), _EPS)
                adjusted.append(logp[i] + np.log(target) - np.log(train))
            adjusted = np.asarray(adjusted, dtype=np.float64)
            adjusted = adjusted - np.max(adjusted)
            expv = np.exp(adjusted)
            probs = expv / expv.sum()

        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx
        top_prob = float(probs[top_idx])
        margin = float(top_prob - float(probs[second_idx]))
        pred_id = int(top_idx)
        if top_prob < self.min_confidence or margin < self.min_margin:
            pred_id = self.neutral_id

        label = self.id2label.get(pred_id, "Neutral")
        return label, top_prob


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
            bar_len = int(pct / 2)
            bar = "#" * bar_len
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
        model_dir = APP_DIR / "data" / "cardiff_sentiment_model_v2"
        if not model_dir.exists():
            model_dir = APP_DIR / "data" / "cardiff_sentiment_model"
        if not model_dir.exists():
            raise FileNotFoundError(
                "Cardiff model not found. Train first with: "
                "python cardiff_sentiment_model.py --output_dir data/cardiff_sentiment_model_v2"
            )

        max_length = int(os.getenv("BERT_MAX_LENGTH", "128"))
        bert_min_conf = float(os.getenv("BERT_MIN_CONFIDENCE", "0.55"))
        bert_min_margin = float(os.getenv("BERT_MIN_MARGIN", "0.10"))
        bert_use_prior = os.getenv("BERT_USE_PRIOR_CORRECTION", "0").strip().lower() in {"1", "true", "yes", "on"}
        bert_target_priors = os.getenv("BERT_TARGET_PRIORS", "")
        bert_train_priors = os.getenv("BERT_TRAIN_PRIORS", "")

        meta_path = model_dir / "model_meta.json"
        use_emote_tags = False
        normalize_twitter = False
        meta_lexicon_path = ""
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            use_emote_tags = bool(meta.get("use_emote_tags", False))
            normalize_twitter = bool(meta.get("normalize_twitter", False))
            meta_lexicon_path = str(meta.get("emote_lexicon_path", ""))

        lexicon_path = os.getenv("BERT_EMOTE_LEXICON", "").strip()
        if not lexicon_path:
            lexicon_path = meta_lexicon_path or "twitch_emote_vader_lexicon.txt"

        if lexicon_path and not Path(lexicon_path).exists():
            local_candidate = APP_DIR / lexicon_path
            if local_candidate.exists():
                lexicon_path = str(local_candidate)
            else:
                project_candidate = APP_DIR / "twitch_emote_vader_lexicon.txt"
                if project_candidate.exists():
                    lexicon_path = str(project_candidate)

        classifier = BertSentimentClassifier(
            model_dir=model_dir,
            max_length=max_length,
            use_emote_tags=use_emote_tags,
            emote_lexicon_path=lexicon_path,
            min_confidence=bert_min_conf,
            min_margin=bert_min_margin,
            use_prior_correction=bert_use_prior,
            target_priors_csv=bert_target_priors,
            train_priors_csv=bert_train_priors,
            normalize_twitter=normalize_twitter,
        )
        print(f"   Loaded: {model_dir}")
        print(f"   Device: {classifier.device}")
        print(f"   Emote tags: {'enabled' if use_emote_tags else 'disabled'}")
        print(f"   Twitter normalization: {'enabled' if normalize_twitter else 'disabled'}")
        print(f"   Neutral gate: max_prob<{bert_min_conf:.2f} or margin<{bert_min_margin:.2f}")
        print(f"   Prior correction: {'enabled' if classifier.use_prior_correction else 'disabled'}")

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
