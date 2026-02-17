import asyncio
import json
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional, Set

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from display_bert import BertSentimentClassifier
from twitch_listener import SimpleTwitchChatListener

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

app = FastAPI(title="Twitch Sentiment Web Chat")

web_dir = APP_DIR / "web"
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

listener: Optional[SimpleTwitchChatListener] = None
listener_task: Optional[asyncio.Task] = None
classifier: Optional[BertSentimentClassifier] = None
connected_clients: Set[WebSocket] = set()
ROLLING_WINDOW_SIZE = 50
recent_messages = deque(maxlen=ROLLING_WINDOW_SIZE)
stats = {
    "running": False,
    "channel": "",
    "total": 0,
    "started_at": 0.0,
    "sentiments": Counter(),
}
_clients_lock = asyncio.Lock()


class StartRequest(BaseModel):
    channel: str


def resolve_model_dir() -> Path:
    env_model_dir = os.getenv("CARDIFF_MODEL_DIR", "").strip()
    if env_model_dir:
        raw = Path(env_model_dir)
        for p in [raw, APP_DIR / raw]:
            if p.exists():
                return p

    candidates = [
        APP_DIR / "data" / "cardiff_sentiment_model",
        APP_DIR / "data" / "cardiff_sentiment_model_v2",
        APP_DIR.parent / "data" / "cardiff_sentiment_model",
        APP_DIR.parent / "data" / "cardiff_sentiment_model_v2",
    ]
    for c in candidates:
        if c.exists():
            return c

    return candidates[0]


def load_classifier() -> BertSentimentClassifier:
    model_dir = resolve_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(
            "Cardiff model not found. Train first with: "
            "python cardiff_sentiment_model.py --output_dir data/cardiff_sentiment_model"
        )

    max_length = int(os.getenv("BERT_MAX_LENGTH", "128"))
    min_conf = float(os.getenv("BERT_MIN_CONFIDENCE", "0.55"))
    min_margin = float(os.getenv("BERT_MIN_MARGIN", "0.10"))
    use_prior = os.getenv("BERT_USE_PRIOR_CORRECTION", "0").strip().lower() in {"1", "true", "yes", "on"}
    target_priors = os.getenv("BERT_TARGET_PRIORS", "")
    train_priors = os.getenv("BERT_TRAIN_PRIORS", "")

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

    lexicon_path = os.getenv("BERT_EMOTE_LEXICON", "").strip() or meta_lexicon_path or "twitch_emote_vader_lexicon.txt"
    lexicon_candidate = Path(lexicon_path)
    if not lexicon_candidate.exists():
        for candidate in [APP_DIR / lexicon_path, APP_DIR.parent / lexicon_path, APP_DIR / "twitch_emote_vader_lexicon.txt"]:
            if candidate.exists():
                lexicon_path = str(candidate)
                break

    return BertSentimentClassifier(
        model_dir=model_dir,
        max_length=max_length,
        use_emote_tags=use_emote_tags,
        emote_lexicon_path=lexicon_path,
        min_confidence=min_conf,
        min_margin=min_margin,
        use_prior_correction=use_prior,
        target_priors_csv=target_priors,
        train_priors_csv=train_priors,
        normalize_twitter=normalize_twitter,
    )


async def broadcast(payload: dict) -> None:
    dead = []
    async with _clients_lock:
        for ws in connected_clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            connected_clients.discard(ws)


async def on_twitch_message(msg_context):
    global stats, classifier, recent_messages
    if classifier is None:
        return
    sentiment, confidence = classifier.predict(msg_context.text)

    stats["total"] += 1
    stats["sentiments"][sentiment] += 1
    recent_messages.append((float(msg_context.timestamp), sentiment))

    payload = {
        "type": "message",
        "timestamp": msg_context.timestamp,
        "username": msg_context.username,
        "username_color": msg_context.username_color or "",
        "text": msg_context.text,
        "emotes": msg_context.emotes or [],
        "sentiment": sentiment,
        "confidence": confidence,
        "channel": stats["channel"],
    }
    await broadcast(payload)
    window_count = len(recent_messages)
    window_counts = Counter([s for _, s in recent_messages])
    mps = 0.0
    if window_count >= 2:
        first_ts = recent_messages[0][0]
        last_ts = recent_messages[-1][0]
        span = max(0.001, last_ts - first_ts)
        mps = window_count / span

    await broadcast(
        {
            "type": "stats",
            "total": stats["total"],
            "window_size": ROLLING_WINDOW_SIZE,
            "window_count": window_count,
            "messages_per_second": mps,
            "sentiments": {
                "Positive": window_counts["Positive"],
                "Neutral": window_counts["Neutral"],
                "Negative": window_counts["Negative"],
            },
            "pct": {
                "Positive": (window_counts["Positive"] / window_count * 100.0 if window_count else 0.0),
                "Neutral": (window_counts["Neutral"] / window_count * 100.0 if window_count else 0.0),
                "Negative": (window_counts["Negative"] / window_count * 100.0 if window_count else 0.0),
            },
        }
    )


@app.on_event("startup")
async def startup_event():
    global classifier
    classifier = load_classifier()


@app.get("/")
async def root():
    return FileResponse(str(web_dir / "chat.html"))


@app.get("/api/status")
async def status():
    window_count = len(recent_messages)
    window_counts = Counter([s for _, s in recent_messages])
    mps = 0.0
    if window_count >= 2:
        first_ts = recent_messages[0][0]
        last_ts = recent_messages[-1][0]
        span = max(0.001, last_ts - first_ts)
        mps = window_count / span
    return {
        "running": stats["running"],
        "channel": stats["channel"],
        "total": stats["total"],
        "started_at": stats["started_at"],
        "model_dir": str(resolve_model_dir()),
        "window_size": ROLLING_WINDOW_SIZE,
        "window_count": window_count,
        "messages_per_second": mps,
        "pct": {
            "Positive": (window_counts["Positive"] / window_count * 100.0 if window_count else 0.0),
            "Neutral": (window_counts["Neutral"] / window_count * 100.0 if window_count else 0.0),
            "Negative": (window_counts["Negative"] / window_count * 100.0 if window_count else 0.0),
        },
    }


@app.post("/api/start")
async def start(req: StartRequest):
    global listener, listener_task, stats, recent_messages

    if not BOT_TOKEN:
        raise HTTPException(status_code=400, detail="Missing TWITCH_BOT_TOKEN in .env")
    channel = req.channel.strip().lower()
    if not channel:
        raise HTTPException(status_code=400, detail="Channel is required")

    if stats["running"]:
        if stats["channel"] == channel:
            return {"ok": True, "channel": channel, "message": f"Already running on #{channel}"}
        await stop()

    bot_token = BOT_TOKEN.strip()
    if not bot_token.lower().startswith("oauth:"):
        bot_token = f"oauth:{bot_token}"

    listener = SimpleTwitchChatListener(
        channel=channel,
        bot_token=bot_token,
        nickname=BOT_NICK,
        on_message_callback=on_twitch_message,
    )
    listener_task = asyncio.create_task(listener.start())

    stats = {
        "running": True,
        "channel": channel,
        "total": 0,
        "started_at": time.time(),
        "sentiments": Counter(),
    }
    recent_messages.clear()
    await broadcast({"type": "system", "text": f"Connected to #{channel}"})
    return {"ok": True, "channel": channel}


@app.post("/api/stop")
async def stop():
    global listener, listener_task, stats, recent_messages
    if listener is not None:
        await listener.stop()
    if listener_task is not None:
        try:
            await asyncio.wait_for(listener_task, timeout=2.0)
        except Exception:
            listener_task.cancel()
    listener = None
    listener_task = None
    was_channel = stats["channel"]
    stats["running"] = False
    stats["channel"] = ""
    recent_messages.clear()
    await broadcast({"type": "system", "text": f"Stopped listener for #{was_channel}"})
    return {"ok": True}


@app.websocket("/ws/messages")
async def websocket_messages(ws: WebSocket):
    await ws.accept()
    async with _clients_lock:
        connected_clients.add(ws)
    try:
        await ws.send_json(
            {
                "type": "status",
                "running": stats["running"],
                "channel": stats["channel"],
                "total": stats["total"],
                "model_dir": str(resolve_model_dir()),
                "window_size": ROLLING_WINDOW_SIZE,
            }
        )
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _clients_lock:
            connected_clients.discard(ws)
