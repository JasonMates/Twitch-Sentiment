import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MsgContext:
    """Metadata extracted from a Twitch msg."""

    message_id: str
    text: str
    username: str
    timestamp: float
    channel: str
    game: str = "Unknown"
    stream_start_time: float = 0
    messages_per_second: float = 0
    velocity_ratio: float = 0
    acceleration: float = 0
    has_emotes: bool = False
    emotes: List[str] = None
    emote_count: int = 0
    has_caps_lock: bool = False
    caps_ratio: float = 0
    punctuation_count: int = 0
    message_length: int = 0
    username_color: Optional[str] = None
    is_moderator: bool = False
    is_subscriber: bool = False
    badges: List[str] = None
    sentiment: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if self.emotes is None:
            self.emotes = []
        if self.badges is None:
            self.badges = []


class ChatListener:
    """
    Connect to Twitch IRC directly.
    """

    def __init__(
        self,
        channel: str,
        bot_token: str,
        nickname: str = "StreamAnalysisBot",
        on_msg: Optional[Callable] = None,
        on_message_callback: Optional[Callable] = None,
    ):
        self.channel = channel.lower()
        self.bot_token = bot_token
        self.nickname = nickname
        self.on_msg = on_msg if on_msg is not None else on_message_callback
        self.on_message_callback = self.on_msg

        self.reader = None
        self.writer = None
        self.is_connected = False
        self.is_running = False

        self.msg_q = asyncio.Queue()
        self.message_queue = self.msg_q
        self.msg_times = deque(maxlen=1000)
        self.message_times = self.msg_times

        logger.info("ChatListener initialized for channel: %s", channel)

    async def _connect(self) -> None:
        """Connect to Twitch IRC."""
        try:
            host = "irc.chat.twitch.tv"
            port = 6667

            logger.info("Connecting to %s:%s", host, port)
            self.reader, self.writer = await asyncio.open_connection(host, port)
            logger.info("Socket connected, authenticating...")

            await self._send(f"PASS {self.bot_token}")
            await self._send(f"NICK {self.nickname}")
            await self._send(f"USER {self.nickname} 8 * :{self.nickname}")
            await self._send("CAP REQ :twitch.tv/membership twitch.tv/tags twitch.tv/commands")

            await asyncio.sleep(1)
            await self._send(f"JOIN #{self.channel}")
            logger.info("Joined #%s", self.channel)
            self.is_connected = True
        except Exception as e:
            logger.error("Connection failed: %s", e)
            self.is_connected = False
            raise

    async def _send(self, msg: str) -> None:
        if not self.writer:
            return
        try:
            self.writer.write((msg + "\r\n").encode())
            await self.writer.drain()
            logger.debug("IRC sent: %s", msg)
        except Exception as e:
            logger.error("Send failed: %s", e)

    async def _recv(self, buf_size: int = 4096) -> Optional[str]:
        if not self.reader:
            return None
        try:
            data = await self.reader.read(buf_size)
            if data:
                return data.decode("utf-8", errors="ignore")
            return None
        except Exception as e:
            logger.error("Receive failed: %s", e)
            return None

    async def _parse_line(self, line: str) -> Optional[MsgContext]:
        if not line or "PRIVMSG" not in line:
            return None

        parts = line.split(" :")
        if len(parts) < 3:
            return None

        tags_str = parts[0]
        msg_text = " ".join(parts[2:])

        tags: Dict[str, str] = {}
        if tags_str.startswith("@"):
            tags_str = tags_str[1:]
            for tag in tags_str.split(";"):
                if "=" in tag:
                    k, v = tag.split("=", 1)
                    tags[k] = v

        username = tags.get("display-name", "unknown")
        msg_id = tags.get("id", str(time.time()))
        ts = int(tags.get("tmi-sent-ts", int(time.time() * 1000))) / 1000

        if username.lower() == self.nickname.lower():
            return None

        emotes = self._extract_emotes(msg_text)
        style = self._text_style(msg_text)
        self.msg_times.append(ts)
        msg_rate = self._msg_rate()

        return MsgContext(
            message_id=msg_id,
            text=msg_text,
            username=username,
            timestamp=ts,
            channel=self.channel,
            game="Unknown",
            messages_per_second=msg_rate,
            velocity_ratio=1.0,
            has_emotes=len(emotes) > 0,
            emotes=emotes,
            emote_count=len(emotes),
            has_caps_lock=style["has_caps"],
            caps_ratio=style["caps_ratio"],
            punctuation_count=style["punctuation_count"],
            message_length=style["length"],
            username_color=tags.get("color"),
            is_moderator="moderator" in tags.get("badges", ""),
            is_subscriber=int(tags.get("subscriber", 0)) == 1,
            badges=tags.get("badges", "").split(","),
        )

    async def _parse_msg(self, raw_line: str) -> Optional[MsgContext]:
        return await self._parse_line(raw_line)

    async def _parse_message(self, raw_line: str) -> Optional[MsgContext]:
        return await self._parse_line(raw_line)

    def _extract_emotes(self, text: str) -> List[str]:
        common_emotes = {
            "POGGERS",
            "PogU",
            "OMEGALUL",
            "LUL",
            "KEKW",
            "Sadge",
            "ResidentSleeper",
            "LULW",
            "Pog",
            "AYAYA",
            "FeelsGoodMan",
            "FeelsStrongMan",
            "OMEGADOWN",
            "WeirdChamp",
            "MONKAS",
            "MonkaS",
            "MONKA",
            "Clueless",
            "BatChest",
            "WICKED",
            "BASED",
            "ICANT",
            "WAYTOODANK",
            "OMEGABRUH",
        }
        return [e for e in common_emotes if e in text]

    def _text_style(self, text: str) -> Dict[str, float]:
        alpha = sum(1 for c in text if c.isalpha())
        caps = sum(1 for c in text if c.isupper())
        return {
            "has_caps": any(c.isupper() for c in text),
            "caps_ratio": caps / alpha if alpha > 0 else 0,
            "punctuation_count": sum(1 for c in text if c in "!?.,;"),
            "length": len(text),
        }

    def _analyze_text_style(self, text: str) -> Dict[str, float]:
        return self._text_style(text)

    def _msg_rate(self) -> float:
        if not self.msg_times or len(self.msg_times) < 2:
            return 0
        span = self.msg_times[-1] - self.msg_times[0]
        if span <= 0:
            return 0
        return len(self.msg_times) / span

    def _calculate_velocity(self) -> float:
        return self._msg_rate()

    async def start(self) -> None:
        self.is_running = True
        try:
            await self._connect()
            logger.info("Connected! Listening to #%s", self.channel)

            buf = ""
            errs = 0

            while self.is_running:
                try:
                    data = await asyncio.wait_for(self._recv(), timeout=30)
                    if not data:
                        logger.warning("Connection closed by server (no data)")
                        break

                    errs = 0
                    buf += data
                    lines = buf.split("\r\n")
                    buf = lines[-1]

                    for line in lines[:-1]:
                        if not line:
                            continue
                        if line.startswith("PING"):
                            await self._send("PONG :tmi.twitch.tv")
                            continue
                        if any(code in line for code in ["001", "002", "003", "004", "375", "376", "CAP", "NOTICE"]):
                            continue

                        msg = await self._parse_line(line)
                        if msg is None:
                            continue

                        await self.msg_q.put(msg)
                        if self.on_msg:
                            await self.on_msg(msg)

                except asyncio.TimeoutError:
                    logger.debug("Receive timeout (normal - waiting for msgs)")
                    continue
                except TypeError as e:
                    if "NoneType" not in str(e):
                        errs += 1
                        logger.error("Error in receive loop: %s (error #%s)", e, errs)
                        if errs > 3:
                            logger.error("Too many consecutive errors, stopping")
                            break
                    await asyncio.sleep(0.01)
                except Exception as e:
                    errs += 1
                    logger.error("Error in receive loop: %s (error #%s)", e, errs)
                    if errs > 3:
                        logger.error("Too many consecutive errors, stopping")
                        break
                    await asyncio.sleep(0.5)
        except Exception as e:
            logger.error("Connection error: %s", e)
        finally:
            await self.stop()

    async def stop(self) -> None:
        self.is_running = False
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
        logger.info("Listener stopped")

    async def get_msg(self) -> Optional[MsgContext]:
        try:
            return await asyncio.wait_for(self.msg_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def get_message(self) -> Optional[MsgContext]:
        return await self.get_msg()


async def example_handler(msg: MsgContext) -> None:
    print(f"[{msg.username}] {msg.text}")


# Compatibility aliases
SimpleTwitchChatListener = ChatListener
MessageContext = MsgContext


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
