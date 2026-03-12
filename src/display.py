import asyncio
import logging
import sys
import signal
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from collections import Counter
import os
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# import listener and hybrid classifier
try:
    from src.deployment.twitch_listener import SimpleTwitchChatListener
    from src.deployment.realtime_analyzer import HybridClassifier
except ImportError:
    from twitch_listener import SimpleTwitchChatListener
    from realtime_analyzer import HybridClassifier

# configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# environment variables
load_dotenv(project_root / '.env')
BOT_TOKEN = os.getenv('TWITCH_BOT_TOKEN')
CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
BOT_NICK = os.getenv('TWITCH_BOT_NICK', 'StreamAnalysisBot')

# global references
listener = None
classifier = None
stats = {
    'total': 0,
    'sentiments': Counter(),
    'start_time': time.time(),
}


def signal_handler(sig, frame):
    """handle Ctrl+C"""
    global listener
    if listener:
        asyncio.create_task(listener.stop())


def get_sentiment_color(sentiment):
    """Get ANSI color code for sentiment"""
    colors = {
        'Positive': '\033[92m',  # Green
        'Negative': '\033[91m',  # Red
        'Neutral': '\033[93m',  # Yellow
    }
    return colors.get(sentiment, '\033[0m')


def normalize_bot_token(token: str) -> str:
    value = (token or "").strip()
    if value and not value.lower().startswith("oauth:"):
        value = f"oauth:{value}"
    return value


async def handle_message(msg_context):
    """display message with sentiment"""
    global classifier, stats

    # classify sentiment
    sentiment, confidence = classifier.predict(msg_context.text)

    # update stats
    stats['total'] += 1
    stats['sentiments'][sentiment] += 1

    # format timestamp
    dt = datetime.fromtimestamp(msg_context.timestamp)
    time_str = dt.strftime('%H:%M:%S')
    # Get color
    color = get_sentiment_color(sentiment)
    reset = '\033[0m'

    # format confidence as percentage
    conf_pct = f"{confidence * 100:.0f}%"

    # format emotes
    emote_display = ''
    if msg_context.emotes:
        emotes_str = ', '.join(msg_context.emotes[:3])
        if len(msg_context.emotes) > 3:
            emotes_str += f' +{len(msg_context.emotes) - 3} more'
        emote_display = f" [{emotes_str}]"

    # Display message
    print(f"{time_str} {color}{sentiment:13s}{reset} ({conf_pct:4s}) | "
          f"{msg_context.username:20s} | {msg_context.text[:70]}{emote_display}")

    # Show periodic stats (every 50 messages)
    if stats['total'] % 50 == 0:
        print_stats()


def print_stats():
    """print statistics"""
    print("\n" + "-" * 100)
    elapsed = time.time() - stats['start_time']
    mins, secs = divmod(int(elapsed), 60)
    rate = stats['total'] / elapsed if elapsed > 0 else 0

    print(f"Stats: {stats['total']} messages | {mins}m {secs}s | {rate:.1f} msg/s")

    if stats['total'] > 0:
        total = sum(stats['sentiments'].values())
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            count = stats['sentiments'][sentiment]
            pct = (count / total * 100) if total > 0 else 0
            color = get_sentiment_color(sentiment)
            reset = '\033[0m'

            # create bar
            bar_len = int(pct / 2)
            bar = "#" * bar_len

            print(f"  {color}{sentiment:13s}{reset}: {bar} {count:3d} ({pct:4.1f}%)")

    print("-" * 100 + "\n")


async def main():
    """run the sentiment display"""
    global listener, classifier

    print("\n" + "=" * 50)
    print("TWITCH CHAT SENTIMENT ANALYZER")
    print("=" * 50)
    print()

    # get channel from user
    while True:
        channel = input("Enter Twitch channel name: ").strip().lower()
        if channel:
            break
        print("Channel name cannot be empty. Please try again.")

    print()

    # validate credentials
    if not BOT_TOKEN:
        print("\nERROR: Missing TWITCH_BOT_TOKEN!")
        print("\nCreate a .env file with:")
        print("TWITCH_BOT_TOKEN=your_token")
        return

    try:
        print(f"[1/3] Loading sentiment classifier...")

        # load VADER emote lexicon
        emote_lexicon_path = project_root / 'src' / 'deployment' / 'twitch_emote_vader_lexicon.txt'
        if not emote_lexicon_path.exists():
            emote_lexicon_path = Path(__file__).parent / 'twitch_emote_vader_lexicon.txt'

        # Init Hybrid Classifier
        if emote_lexicon_path.exists():
            classifier = HybridClassifier(
                emote_lexicon_path=str(emote_lexicon_path),
                use_vader=True
            )
            print(f"   Loaded emote lexicon from: {emote_lexicon_path.name}")
        else:
            classifier = HybridClassifier(use_vader=True)
            print(f"   Emote lexicon not found - using basic mode")

        # check VADER status
        if classifier.use_vader and classifier.vader:
            print(f"   VADER text analysis enabled")
        else:
            print(f"VADER not available - using emote-only mode")
            print(f"      Install with: pip install vaderSentiment")

        print(f"[2/3] Connecting to Twitch IRC...")
        listener = SimpleTwitchChatListener(
            channel=channel,
            bot_token=normalize_bot_token(BOT_TOKEN),
            nickname=BOT_NICK,
            on_message_callback=handle_message,
        )

        print(f"[3/3] Starting chat listener...\n")
        print("Connected! Analyzing messages...\n")

        # signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)

        # start listening
        await listener.start()

    except asyncio.CancelledError:
        print("\n\n  " + "=" * 96)
        print("    Shutting down...")

        # final stats
        if stats['total'] > 0:
            print("\n   FINAL STATISTICS")
            print("  " + "=" * 96)
            print_stats()

        print("\n  Shutdown complete\n")

    except Exception as e:
        print(f"\n  Error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())


