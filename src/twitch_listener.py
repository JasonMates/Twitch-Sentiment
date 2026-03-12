import asyncio
import logging
import time
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Dict, Callable
import re

logger = logging.getLogger(__name__)


# Data Classes
@dataclass
class MessageContext:
    """Metadata extracted from a Twitch message"""
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

    def __post_init__(self):
        if self.emotes is None:
            self.emotes = []
        if self.badges is None:
            self.badges = []


# Simple Direct IRC
class SimpleTwitchChatListener:
    """
    Connect to Twitch IRC directly
    Much simpler and more reliable for just listening to chat messages.
    """

    def __init__(
        self,
        channel: str,
        bot_token: str,
        nickname: str = 'StreamAnalysisBot',
        on_message_callback: Optional[Callable] = None,
    ):
        self.channel = channel.lower()
        self.bot_token = bot_token
        self.nickname = nickname
        self.on_message_callback = on_message_callback

        # IRC connection (using asyncio reader/writer for windows)
        self.reader = None
        self.writer = None
        self.is_connected = False
        self.is_running = False

        # message queue
        self.message_queue = asyncio.Queue()

        # velocity tracking
        self.message_times = deque(maxlen=1000)

        logger.info(f"SimpleTwitchChatListener initialized for channel: {channel}")

    async def _connect(self):
        """connect to Twitch IRC using asyncio"""
        try:
            host = 'irc.chat.twitch.tv'
            port = 6667

            logger.info(f"Connecting to {host}:{port}")

            # use asyncio.open_connection
            self.reader, self.writer = await asyncio.open_connection(host, port)
            logger.info("Socket connected, authenticating...")

            # Send authentication commands
            await self._send(f"PASS {self.bot_token}")
            await self._send(f"NICK {self.nickname}")
            await self._send(f"USER {self.nickname} 8 * :{self.nickname}")

            # request capabilities
            await self._send("CAP REQ :twitch.tv/membership twitch.tv/tags twitch.tv/commands")

            # wait a moment for server to process auth
            await asyncio.sleep(1)

            # join channel
            await self._send(f"JOIN #{self.channel}")
            logger.info(f"Joined #{self.channel}")

            self.is_connected = True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
            raise

    async def _send(self, message: str):
        """send a message to IRC server"""
        if not self.writer:
            return

        try:
            self.writer.write((message + "\r\n").encode())
            await self.writer.drain()
            logger.debug(f"IRC sent: {message}")
        except Exception as e:
            logger.error(f"Send failed: {e}")

    async def _receive(self, buffer_size: int = 4096) -> Optional[str]:
        """receive data from IRC server"""
        if not self.reader:
            return None

        try:
            data = await self.reader.read(buffer_size)
            if data:
                return data.decode('utf-8', errors='ignore')
            return None
        except Exception as e:
            logger.error(f"Receive failed: {e}")
            return None

    async def _parse_message(self, raw_line: str) -> Optional[MessageContext]:
        """parse IRC message and extract metadata"""
        try:
            if not raw_line or 'PRIVMSG' not in raw_line:
                return None

            # split tags from message
            parts = raw_line.split(' :')
            if len(parts) < 3:
                return None

            tags_str = parts[0]
            message_text = ' '.join(parts[2:])

            # parse tags
            tags = {}
            if tags_str.startswith('@'):
                tags_str = tags_str[1:]  # Remove @
                for tag in tags_str.split(';'):
                    if '=' in tag:
                        key, val = tag.split('=', 1)
                        tags[key] = val

            # extract key fields
            username = tags.get('display-name', 'unknown')
            message_id = tags.get('id', str(time.time()))
            timestamp = int(tags.get('tmi-sent-ts', int(time.time() * 1000))) / 1000

            # skeeep bot messages
            if username.lower() == self.nickname.lower():
                return None

            # extract emotes
            emotes = self._extract_emotes(message_text)

            # analyze text style
            style = self._analyze_text_style(message_text)

            # track velocity
            self.message_times.append(timestamp)
            velocity = self._calculate_velocity()

            # create message context
            msg_context = MessageContext(
                message_id=message_id,
                text=message_text,
                username=username,
                timestamp=timestamp,
                channel=self.channel,
                game='Unknown',  # We'd need Twitch API for this
                messages_per_second=velocity,
                velocity_ratio=1.0,  # Would need baseline
                has_emotes=len(emotes) > 0,
                emotes=emotes,
                emote_count=len(emotes),
                has_caps_lock=style['has_caps'],
                caps_ratio=style['caps_ratio'],
                punctuation_count=style['punctuation_count'],
                message_length=style['length'],
                username_color=tags.get('color'),
                is_moderator='moderator' in tags.get('badges', ''),
                is_subscriber=int(tags.get('subscriber', 0)) == 1,
                badges=tags.get('badges', '').split(','),
            )

            return msg_context

        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return None

    def _extract_emotes(self, text: str) -> List[str]:
        """extract Twitch emotes from message"""
        common_emotes = {
            'POGGERS', 'PogU', 'OMEGALUL', 'LUL', 'KEKW', 'Sadge',
            'ResidentSleeper', 'LULW', 'Pog', 'AYAYA',
            'FeelsGoodMan', 'FeelsStrongMan', 'OMEGADOWN', 'WeirdChamp',
            'MONKAS', 'MonkaS', 'MONKA', 'Clueless', 'BatChest',
            'WICKED', 'BASED', 'ICANT', 'WAYTOODANK', 'OMEGABRUH',
        }

        emotes = [e for e in common_emotes if e in text]
        return emotes

    def _analyze_text_style(self, text: str) -> Dict:
        """analyze text styling"""
        alpha_chars = sum(1 for c in text if c.isalpha())
        caps_chars = sum(1 for c in text if c.isupper())

        return {
            'has_caps': any(c.isupper() for c in text),
            'caps_ratio': caps_chars / alpha_chars if alpha_chars > 0 else 0,
            'punctuation_count': sum(1 for c in text if c in '!?.,;'),
            'length': len(text),
        }

    def _calculate_velocity(self) -> float:
        """calculate messages per second"""
        if not self.message_times or len(self.message_times) < 2:
            return 0

        time_span = self.message_times[-1] - self.message_times[0]
        if time_span <= 0:
            return 0

        return len(self.message_times) / time_span

    async def start(self):
        """start listening to chat"""
        self.is_running = True

        try:
            await self._connect()
            logger.info(f"Connected! Listening to #{self.channel}")

            buffer = ""
            consecutive_errors = 0

            while self.is_running:
                try:
                    # receive data from IRC
                    data = await asyncio.wait_for(self._receive(), timeout=30)

                    if not data:
                        logger.warning("Connection closed by server (no data)")
                        break

                    consecutive_errors = 0  # reset error counter on successful receive

                    # process incoming data
                    buffer += data
                    lines = buffer.split('\r\n')
                    buffer = lines[-1]  # Keep incomplete line in buffer

                    # process complete lines
                    for line in lines[:-1]:
                        if not line:
                            continue

                        # handle ping !!!CRITICAL!!! for keeping connection alive
                        if line.startswith('PING'):
                            await self._send('PONG :tmi.twitch.tv')
                            continue

                        # skip server messages we don't care about
                        if any(code in line for code in ['001', '002', '003', '004', '375', '376', 'CAP', 'NOTICE']):
                            continue

                        # parse message
                        msg_context = await self._parse_message(line)
                        if msg_context:
                            await self.message_queue.put(msg_context)

                            if self.on_message_callback:
                                await self.on_message_callback(msg_context)

                except asyncio.TimeoutError:
                    # timeout is normal
                    logger.debug("Receive timeout (normal - waiting for messages)")
                    continue

                except TypeError as e:
                    # ignore "object NoneType can't be used in 'await' expression"
                    # this is normal when parsing non-PRIVMSG lines that return None
                    if "NoneType" not in str(e):
                        consecutive_errors += 1
                        logger.error(f"Error in receive loop: {e} (error #{consecutive_errors})")
                        if consecutive_errors > 3:
                            logger.error("Too many consecutive errors, stopping")
                            break
                    await asyncio.sleep(0.01)

                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error in receive loop: {e} (error #{consecutive_errors})")

                    # if we get too many errors in a row, stop
                    if consecutive_errors > 3:
                        logger.error("Too many consecutive errors, stopping")
                        break

                    # wait a moment before trying again
                    await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """stop listening"""
        self.is_running = False
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except:
                pass
        logger.info("Listener stopped")

    async def get_message(self) -> Optional[MessageContext]:
        """get next message from queue"""
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None


# example usage
async def example_handler(msg: MessageContext):
    print(f"[{msg.username}] {msg.text}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)