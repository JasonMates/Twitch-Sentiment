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

from display_bert import SentModel, get_model_src
from twitch_listener import ChatListener

APP_DIR = Path(__file__).resolve().parent


def _load_env() -> None:
    for base in [APP_DIR, *APP_DIR.parents]:
        env_path = base / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return
    load_dotenv()


_load_env()

BOT_TOKEN = os.getenv("TWITCH_BOT_TOKEN")
BOT_NICK = os.getenv("TWITCH_BOT_NICK", "StreamAnalysisBot")

app = FastAPI(title="Twitch Sentiment Web Chat")

web_dir = APP_DIR / "web"
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

listener: Optional[ChatListener] = None
listen_task: Optional[asyncio.Task] = None
clf: Optional[SentModel] = None
clients: Set[WebSocket] = set()
WIN = 50
recent_msgs = deque(maxlen=WIN)
stats = {
    "running": False,
    "channel": "",
    "total": 0,
    "started_at": 0.0,
    "sentiments": Counter(),
}
_ws_lock = asyncio.Lock()


class StartReq(BaseModel):
    channel: str


def load_clf() -> SentModel:
    model_source, model_opts = get_model_src(APP_DIR)
    local_model_dir = Path(model_source)
    has_local_model = local_model_dir.exists()
    if not has_local_model and not os.getenv("MODEL_ID", "").strip():
        raise FileNotFoundError(
            "Local Cardiff model not found. Set MODEL_ID=JDMates/TwitchRoBERTaSentiment "
            "or train first with: python cardiff_sentiment_model.py --output_dir data/cardiff_sentiment_model_v2"
        )

    max_len = int(os.getenv("BERT_MAX_LENGTH", "128"))
    min_conf = float(os.getenv("BERT_MIN_CONFIDENCE", "0.55"))
    min_gap = float(os.getenv("BERT_MIN_MARGIN", "0.10"))
    use_prior = os.getenv("BERT_USE_PRIOR_CORRECTION", "0").strip().lower() in {"1", "true", "yes", "on"}
    target_priors = os.getenv("BERT_TARGET_PRIORS", "")
    train_priors = os.getenv("BERT_TRAIN_PRIORS", "")

    meta_path = local_model_dir / "model_meta.json"
    use_emote_tags = False
    normalize_twitter = False
    meta_lexicon_path = ""
    if has_local_model and meta_path.exists():
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

    return SentModel(
        model_src=model_source,
        max_len=max_len,
        use_emote_tags=use_emote_tags,
        emote_lex=lexicon_path,
        min_conf=min_conf,
        min_gap=min_gap,
        use_prior=use_prior,
        target_priors=target_priors,
        train_priors=train_priors,
        norm_twitter=normalize_twitter,
        rev=model_opts.get("revision"),
        token=model_opts.get("token"),
        local_only=False,
    )


async def send_all(payload: dict) -> None:
    dead = []
    async with _ws_lock:
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)


async def on_twitch_msg(msg_context):
    global stats, clf, recent_msgs
    if clf is None:
        return
    sentiment, confidence = clf.predict(msg_context.text)

    stats["total"] += 1
    stats["sentiments"][sentiment] += 1
    recent_msgs.append((float(msg_context.timestamp), sentiment))

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
    await send_all(payload)
    win_n = len(recent_msgs)
    win_counts = Counter([s for _, s in recent_msgs])
    mps = 0.0
    if win_n >= 2:
        first_ts = recent_msgs[0][0]
        last_ts = recent_msgs[-1][0]
        span = max(0.001, last_ts - first_ts)
        mps = win_n / span

    await send_all(
        {
            "type": "stats",
            "total": stats["total"],
            "window_size": WIN,
            "window_count": win_n,
            "messages_per_second": mps,
            "sentiments": {
                "Positive": win_counts["Positive"],
                "Neutral": win_counts["Neutral"],
                "Negative": win_counts["Negative"],
            },
            "pct": {
                "Positive": (win_counts["Positive"] / win_n * 100.0 if win_n else 0.0),
                "Neutral": (win_counts["Neutral"] / win_n * 100.0 if win_n else 0.0),
                "Negative": (win_counts["Negative"] / win_n * 100.0 if win_n else 0.0),
            },
        }
    )


@app.on_event("startup")
async def on_start():
    global clf
    clf = load_clf()


@app.get("/")
async def root():
    return FileResponse(str(web_dir / "chat.html"))


@app.get("/api/status")
async def status():
    model_source, _ = get_model_src(APP_DIR)
    win_n = len(recent_msgs)
    win_counts = Counter([s for _, s in recent_msgs])
    mps = 0.0
    if win_n >= 2:
        first_ts = recent_msgs[0][0]
        last_ts = recent_msgs[-1][0]
        span = max(0.001, last_ts - first_ts)
        mps = win_n / span
    return {
        "running": stats["running"],
        "channel": stats["channel"],
        "total": stats["total"],
        "started_at": stats["started_at"],
        "model_source": str(model_source),
        "window_size": WIN,
        "window_count": win_n,
        "messages_per_second": mps,
        "pct": {
            "Positive": (win_counts["Positive"] / win_n * 100.0 if win_n else 0.0),
            "Neutral": (win_counts["Neutral"] / win_n * 100.0 if win_n else 0.0),
            "Negative": (win_counts["Negative"] / win_n * 100.0 if win_n else 0.0),
        },
    }


@app.post("/api/start")
async def start(req: StartReq):
    global listener, listen_task, stats, recent_msgs

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

    listener = ChatListener(
        channel=channel,
        bot_token=bot_token,
        nickname=BOT_NICK,
        on_msg=on_twitch_msg,
    )
    listen_task = asyncio.create_task(listener.start())

    stats = {
        "running": True,
        "channel": channel,
        "total": 0,
        "started_at": time.time(),
        "sentiments": Counter(),
    }
    recent_msgs.clear()
    await send_all({"type": "system", "text": f"Connected to #{channel}"})
    return {"ok": True, "channel": channel}


@app.post("/api/stop")
async def stop():
    global listener, listen_task, stats, recent_msgs
    if listener is not None:
        await listener.stop()
    if listen_task is not None:
        try:
            await asyncio.wait_for(listen_task, timeout=2.0)
        except Exception:
            listen_task.cancel()
    listener = None
    listen_task = None
    was_channel = stats["channel"]
    stats["running"] = False
    stats["channel"] = ""
    recent_msgs.clear()
    await send_all({"type": "system", "text": f"Stopped listener for #{was_channel}"})
    return {"ok": True}


@app.websocket("/ws/messages")
async def ws_msgs(ws: WebSocket):
    await ws.accept()
    async with _ws_lock:
        clients.add(ws)
    model_source, _ = get_model_src(APP_DIR)
    try:
        await ws.send_json(
            {
                "type": "status",
                "running": stats["running"],
                "channel": stats["channel"],
                "total": stats["total"],
                "model_source": str(model_source),
                "window_size": WIN,
            }
        )
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            clients.discard(ws)


# Compatibility aliases
StartRequest = StartReq
load_classifier = load_clf
broadcast = send_all
on_twitch_message = on_twitch_msg
startup_event = on_start
websocket_messages = ws_msgs
ROLLING_WINDOW_SIZE = WIN
recent_messages = recent_msgs
connected_clients = clients
_clients_lock = _ws_lock

