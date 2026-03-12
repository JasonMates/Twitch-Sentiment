import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# page config
st.set_page_config(page_title="Twitch Sentiment Labeler", layout="centered")

# assignment config
TEAM_MEMBERS = ["Bill", "Jason", "julian"]
TARGET_PER_USER = 200
SOURCE_ID_PREFIX = "SRC-"
SOURCE_CSV_FILENAME = "Twitch_Sentiment_Labels - Sheet1 (5).csv"
VALID_SENTIMENTS = ["Positive", "Negative", "Neutral"]
SENTIMENT_TARGETS = {"Negative": 80, "Neutral": 80, "Positive": 40}


# initialize session state
if 'current_message' not in st.session_state:
    st.session_state.current_message = None
if 'message_index' not in st.session_state:
    st.session_state.message_index = None
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'source_df' not in st.session_state:
    st.session_state.source_df = pd.DataFrame()
if 'source_csv_path' not in st.session_state:
    st.session_state.source_csv_path = ""
if 'sheet_connected' not in st.session_state:
    st.session_state.sheet_connected = False
if 'sheet' not in st.session_state:
    st.session_state.sheet = None
if 'active_labeler' not in st.session_state:
    st.session_state.active_labeler = None
if 'session_labeled_ids' not in st.session_state:
    st.session_state.session_labeled_ids = set()


def normalize_text(value):
    """Normalize message text for duplicate checks."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split()).lower()


@st.cache_data
def load_local_csv_messages():
    """Load source messages from a local CSV in the project root."""
    required_columns = {"message_id", "message", "sentiment", "labeled_by"}

    preferred_path = Path(SOURCE_CSV_FILENAME)
    csv_candidates = []
    if preferred_path.exists():
        csv_candidates.append(preferred_path)
    csv_candidates.extend([path for path in sorted(Path(".").glob("*.csv")) if path != preferred_path])

    for csv_path in csv_candidates:
        try:
            raw_df = pd.read_csv(csv_path)
        except Exception:
            continue

        column_map = {col.strip().lower(): col for col in raw_df.columns}
        if not required_columns.issubset(set(column_map.keys())):
            continue

        source_df = pd.DataFrame({
            "source_message_id": raw_df[column_map["message_id"]].fillna("").astype(str).str.strip(),
            "message": raw_df[column_map["message"]].fillna("").astype(str).str.strip(),
            "source_sentiment": raw_df[column_map["sentiment"]].fillna("").astype(str).str.strip().str.title(),
            "source_labeler": raw_df[column_map["labeled_by"]].fillna("").astype(str).str.strip(),
        })

        source_df = source_df[source_df["message"] != ""]
        source_df = source_df[source_df["source_sentiment"].isin(VALID_SENTIMENTS)]
        source_df = source_df.reset_index(drop=True)
        source_df["source_row"] = source_df.index + 1
        source_df["source_id"] = SOURCE_ID_PREFIX + source_df["source_row"].astype(str)

        return source_df, str(csv_path), None

    return pd.DataFrame(), "", "No local CSV found with columns: message_id, message, sentiment, labeled_by."


def get_user_sheet_state(labels_df, labeler_name):
    """Build per-user label history and overlap progress from Google Sheets."""
    empty_counts = {sentiment: 0 for sentiment in VALID_SENTIMENTS}

    if labels_df.empty or not labeler_name or "labeled_by" not in labels_df.columns:
        return set(), set(), empty_counts, 0

    user_df = labels_df[
        labels_df["labeled_by"].fillna("").astype(str).str.strip().str.lower() == labeler_name.lower()
    ].copy()

    if user_df.empty:
        return set(), set(), empty_counts, 0

    source_ids = set()
    if "message_id" in user_df.columns:
        for raw_id in user_df["message_id"].fillna("").astype(str):
            current_id = raw_id.strip()
            if current_id.startswith(SOURCE_ID_PREFIX):
                source_ids.add(current_id)

    labeled_messages = set()
    if "message" in user_df.columns:
        labeled_messages = {normalize_text(msg) for msg in user_df["message"].fillna("").astype(str)}

    overlap_counts = empty_counts.copy()
    completed_overlap = 0
    if "message_id" in user_df.columns and "sentiment" in user_df.columns:
        overlap_df = user_df[user_df["message_id"].fillna("").astype(str).str.startswith(SOURCE_ID_PREFIX)]
        if not overlap_df.empty:
            sentiment_counts = overlap_df["sentiment"].fillna("").astype(str).str.strip().str.title().value_counts()
            for sentiment in VALID_SENTIMENTS:
                overlap_counts[sentiment] = int(sentiment_counts.get(sentiment, 0))
            completed_overlap = int(sum(overlap_counts.values()))

    return source_ids, labeled_messages, overlap_counts, completed_overlap


def select_next_message(source_df, labels_df, labeler_name, session_labeled_ids):
    """Deterministically select the next eligible message for a user."""
    if source_df.empty:
        return None, "No source messages are loaded."

    source_ids, labeled_messages, overlap_counts, _ = get_user_sheet_state(labels_df, labeler_name)
    session_id_set = set(session_labeled_ids)
    already_labeled_ids = source_ids.union(session_id_set)

    effective_completed = len(already_labeled_ids)
    if effective_completed >= TARGET_PER_USER:
        return None, f"Target reached ({TARGET_PER_USER} labels)."

    # Include unsynced current-session labels when enforcing sentiment quotas.
    unsynced_ids = session_id_set.difference(source_ids)
    if unsynced_ids:
        unsynced_rows = source_df[source_df["source_id"].isin(unsynced_ids)]
        if not unsynced_rows.empty:
            unsynced_counts = unsynced_rows["source_sentiment"].value_counts()
            for sentiment in VALID_SENTIMENTS:
                overlap_counts[sentiment] = overlap_counts.get(sentiment, 0) + int(unsynced_counts.get(sentiment, 0))

    eligible_df = source_df[
        source_df["source_labeler"].fillna("").astype(str).str.lower() != labeler_name.lower()
    ].copy()
    eligible_df = eligible_df[~eligible_df["source_id"].isin(already_labeled_ids)]
    eligible_df = eligible_df[~eligible_df["message"].map(normalize_text).isin(labeled_messages)]
    eligible_df = eligible_df.sort_values("source_row")

    if eligible_df.empty:
        return None, "No remaining eligible messages for this user."

    remaining_targets = {
        sentiment: max(SENTIMENT_TARGETS[sentiment] - overlap_counts.get(sentiment, 0), 0)
        for sentiment in VALID_SENTIMENTS
    }
    sentiment_priority = ["Negative", "Neutral", "Positive"]
    ordered_sentiments = sorted(
        sentiment_priority,
        key=lambda sentiment: (-remaining_targets[sentiment], sentiment_priority.index(sentiment))
    )

    for sentiment in ordered_sentiments:
        if remaining_targets[sentiment] <= 0:
            continue
        bucket_df = eligible_df[eligible_df["source_sentiment"] == sentiment]
        if not bucket_df.empty:
            return bucket_df.iloc[0].to_dict(), None

    # fallback if a target bucket is exhausted but other eligible messages remain
    for sentiment in sentiment_priority:
        bucket_df = eligible_df[eligible_df["source_sentiment"] == sentiment]
        if not bucket_df.empty:
            return bucket_df.iloc[0].to_dict(), None

    return None, "No remaining eligible messages for this user."


# google Sheets functions
@st.cache_resource
def init_google_sheets():
    """Initialize Google Sheets connection"""
    try:
        # Get credentials from Streamlit secrets
        creds_dict = st.secrets["google_sheets"]

        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        # open the spreadsheet
        spreadsheet = client.open('Twitch_Sentiment_Labels')
        sheet = spreadsheet.sheet1

        return sheet, True
    except Exception as e:
        st.error(f"❌ Google Sheets connection failed: {e}")
        return None, False


def load_labels_from_sheet(sheet):
    """Load all labels from Google Sheet"""
    try:
        records = sheet.get_all_records()
        if records:
            return pd.DataFrame(records)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return pd.DataFrame()


def save_label_to_sheet(sheet, message_id, message, sentiment, confidence, labeler_name, timestamp):
    """Save a single label to Google Sheet"""
    try:
        sheet.append_row([
            message_id,
            message,
            sentiment,
            confidence,
            labeler_name,
            timestamp
        ])
        return True
    except Exception as e:
        st.error(f"Error saving to sheet: {e}")
        return False


# title and description
st.title("Twitch Chat Sentiment Labeler")
st.markdown("Label Twitch chat messages by sentiment.")

# sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    selected_labeler = st.selectbox("Labeler:", ["Select user..."] + TEAM_MEMBERS, index=0)
    labeler_name = None if selected_labeler == "Select user..." else selected_labeler

    if st.session_state.active_labeler != labeler_name:
        st.session_state.active_labeler = labeler_name
        st.session_state.current_message = None
        st.session_state.message_index = None
        st.session_state.session_labeled_ids = set()

    st.divider()

    # google Sheets Connection
    st.subheader("📊 Google Sheets Setup")

    if st.button("🔗 Connect to Google Sheets", use_container_width=True):
        with st.spinner("Connecting to Google Sheets..."):
            sheet, connected = init_google_sheets()
            if connected:
                st.session_state.sheet = sheet
                st.session_state.sheet_connected = True
                st.success("✅ Connected to Google Sheets!")
            else:
                st.session_state.sheet_connected = False

    if st.session_state.sheet_connected:
        st.success("✅ Google Sheets connected")

        # show current stats from sheet
        if st.button("📂 Refresh Stats from Sheet", use_container_width=True):
            with st.spinner("Loading labels..."):
                df = load_labels_from_sheet(st.session_state.sheet)
                if not df.empty:
                    if labeler_name:
                        df_user = df[
                            df["labeled_by"].fillna("").astype(str).str.strip().str.lower() == labeler_name.lower()
                        ]
                        st.metric("Your Labels", len(df_user))
                    st.metric("Total Labels", len(df))
    else:
        st.warning("⚠️ Not connected to Google Sheets yet")

    st.divider()

    # load local CSV source
    source_df, source_csv_path, source_error = load_local_csv_messages()
    if source_error:
        st.session_state.dataset_loaded = False
        st.session_state.source_df = pd.DataFrame()
        st.session_state.source_csv_path = ""
        st.error(source_error)
    else:
        st.session_state.dataset_loaded = True
        st.session_state.source_df = source_df
        st.session_state.source_csv_path = source_csv_path
        st.info(f"📄 Source CSV: {Path(source_csv_path).name} ({len(source_df)} messages)")

    st.divider()
    st.subheader("📖 Sentiment Guide")


    with st.expander("Positive"):
        st.write("""
        **Positive but calmer approval / friendliness**
        - Examples (from this dataset): gg, ggs, nice, ty, thanks, thank you, based, wp, awesome, amazing, LOVE YOU
        - Signs: compliments, gratitude, “good/nice” language; supportive wording
        """)

    with st.expander("Negative"):
        st.write("""
        **Negative sentiment (anger, insults, disappointment, complaints)**
        - Examples (from this dataset): wtf, trash, cringe, sucks, hate, moron, fuck, shit, bitch
        - Signs: insults/blame, aggressive profanity, “this sucks” style complaints, accusatory tone, hostile wording
        """)


    with st.expander("Neutral"):
        st.write("""
        **Low-emotion / informational / general chat**
        - Examples (from this dataset): hi, hello, LIVE, uptime, map, tomorrow
        - Signs: greetings, basic questions, observations without strong opinion markers, minimal emotive punctuation
        """)


# main content
if not st.session_state.dataset_loaded:
    st.warning("⚠️ Local CSV source could not be loaded.")
elif not st.session_state.sheet_connected:
    st.warning("⚠️ Click '🔗 Connect to Google Sheets' in the sidebar to sync labels!")
elif not labeler_name:
    st.warning("⚠️ Select your name from the sidebar to continue.")
else:
    labels_df = load_labels_from_sheet(st.session_state.sheet)
    sheet_source_ids, _, overlap_counts, _ = get_user_sheet_state(labels_df, labeler_name)
    effective_completed = len(sheet_source_ids.union(st.session_state.session_labeled_ids))

    col1, col2 = st.columns([2, 1])

    with col1:
        load_disabled = st.session_state.current_message is not None or effective_completed >= TARGET_PER_USER
        if st.button("🔄 Load Next Message", use_container_width=True, disabled=load_disabled):
            next_message, load_error = select_next_message(
                st.session_state.source_df,
                labels_df,
                labeler_name,
                st.session_state.session_labeled_ids,
            )
            if next_message:
                st.session_state.current_message = next_message["message"]
                st.session_state.message_index = next_message["source_id"]
            else:
                st.warning(load_error or "No eligible message available.")

    with col2:
        if st.session_state.current_message:
            st.caption("Submit the current label to continue.")
        else:
            remaining = max(TARGET_PER_USER - effective_completed, 0)
            st.caption(f"Remaining target: {remaining}")

    # Load Twitch API credentials from secrets
    twitch_client_id = st.secrets.get("twitch", {}).get("client_id")
    twitch_token = st.secrets.get("twitch", {}).get("access_token")
    twitch_creator_id = st.secrets.get("twitch", {}).get("twitch_creator_id", "121059319")


    # load BTTV emotes from API
    @st.cache_resource
    def load_bttv_emotes():
        """Load BTTV emotes from the API"""
        import requests

        try:
            # fetch global emotes
            response = requests.get('https://api.betterttv.net/3/emotes/global', timeout=10)
            if response.status_code == 200:
                data = response.json()
                emotes = {}
                for emote in data:
                    emote_code = emote.get('code')
                    emote_id = emote.get('id')
                    image_type = emote.get('imageType', 'png')
                    if emote_code and emote_id:
                        emotes[emote_code] = {
                            'id': emote_id,
                            'type': image_type
                        }
                return emotes if emotes else {}
        except Exception as e:
            st.warning(f"Could not load BTTV emotes: {e}")

        return {}


    @st.cache_resource
    def load_bttv_channel_emotes():
        """Load BTTV emotes for the specified creator channel"""
        import requests

        try:
            response = requests.get(
                f'https://api.betterttv.net/3/cached/users/twitch/{twitch_creator_id}',
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                emotes = {}

                # Load channel emotes
                for emote in data.get('channelEmotes', []):
                    emote_code = emote.get('code')
                    emote_id = emote.get('id')
                    if emote_code and emote_id:
                        emotes[emote_code] = {
                            'id': emote_id,
                            'type': 'gif' if emote.get('animated') else 'png',
                            'source': 'bttv_channel'
                        }

                # Load shared emotes
                for emote in data.get('sharedEmotes', []):
                    emote_code = emote.get('code')
                    emote_id = emote.get('id')
                    if emote_code and emote_id:
                        emotes[emote_code] = {
                            'id': emote_id,
                            'type': 'gif' if emote.get('animated') else 'png',
                            'source': 'bttv_shared'
                        }

                return emotes if emotes else {}
        except Exception as e:
            st.warning(f"Could not load BTTV channel emotes for {twitch_creator_id}: {e}")

        return {}


    @st.cache_resource
    def load_twitch_global_emotes():
        """Load Twitch native global emotes"""
        import requests

        if not twitch_client_id or not twitch_token:
            st.warning("Twitch API credentials not configured in secrets.toml")
            return {}

        try:
            headers = {
                'Authorization': f'Bearer {twitch_token}',
                'Client-ID': twitch_client_id
            }
            response = requests.get(
                'https://api.twitch.tv/helix/chat/emotes/global',
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                emotes = {}

                for emote in data.get('data', []):
                    emote_name = emote.get('name')
                    emote_id = emote.get('id')
                    images = emote.get('images', {})

                    if emote_name and emote_id:
                        # Use 2x image URL from Twitch
                        image_url = images.get('url_2x', '')
                        emotes[emote_name] = {
                            'id': emote_id,
                            'url': image_url,
                            'source': 'twitch_native'
                        }

                return emotes if emotes else {}
        except Exception as e:
            st.warning(f"Could not load Twitch global emotes: {e}")

        return {}


    # Load emotes from all sources
    emote_map_twitch = load_twitch_global_emotes()
    emote_map_bttv_global = load_bttv_emotes()
    emote_map_bttv_channel = load_bttv_channel_emotes()

    # Combine all emotes (Twitch takes priority, then BTTV channel, then BTTV global)
    emote_map_combined = {}
    emote_map_combined.update(emote_map_bttv_global)  # Start with global BTTV
    emote_map_combined.update(emote_map_bttv_channel)  # Override with channel BTTV
    emote_map_combined.update(emote_map_twitch)  # Override with Twitch (highest priority)


    def get_emote_url(emote_data):
        """Get emote image URL from any source (Twitch, BTTV global, or BTTV channel)"""
        try:
            source = emote_data.get('source', 'bttv')

            # Twitch native emotes have direct URLs
            if source == 'twitch_native':
                return emote_data.get('url')

            # BTTV emotes need CDN construction
            emote_id = emote_data.get('id')
            image_type = emote_data.get('type', 'png')

            if emote_id:
                return f"https://cdn.betterttv.net/emote/{emote_id}/2x.{image_type}"
        except Exception as e:
            return None
        return None


    def render_message_with_emotes(text):
        """Render message with emotes from Twitch and BTTV"""
        html = f'<div style="font-size: 18px; line-height: 1.8;">'
        words = text.split()

        for word in words:
            if word in emote_map_combined:
                emote_data = emote_map_combined[word]
                emote_url = get_emote_url(emote_data)
                if emote_url:
                    html += f'<img src="{emote_url}" alt="{word}" style="height: 28px; margin: 0 2px; vertical-align: middle;" onerror="this.style.display=\'none\'">'
                else:
                    html += f'<span style="margin-right: 4px;">{word}</span>'
            else:
                html += f'<span style="margin-right: 4px;">{word}</span>'

        html += '</div>'
        return html


    # display current message
    if st.session_state.current_message:
        st.divider()

        st.markdown("### Current Message")
        message_container = st.container(border=True)
        with message_container:
            st.write(f"**ID:** {st.session_state.message_index}")
            st.caption("Assigned from a different user in the CSV source.")
            # display message with emotes
            emote_html = render_message_with_emotes(st.session_state.current_message)
            st.markdown(emote_html, unsafe_allow_html=True)
            st.caption(f"Original text: {st.session_state.current_message}")

        st.divider()

        st.markdown("### Select Sentiment")

        col1, col2 = st.columns(2)

        with col1:
            sentiment = st.selectbox(
                "Sentiment:",
                ["Select...", "Positive", "Negative", "Neutral"],
                index=0,
                key="sentiment_select"
            )

        with col2:
            confidence = st.selectbox(
                "Confidence:",
                ["Select...", "1 - Very Unsure", "2 - Unsure", "3 - Neutral", "4 - Confident", "5 - Very Confident"],
                index=0,
                key="confidence_select"
            )

        st.divider()

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("✅ Submit Label", use_container_width=True, type="primary"):
                if sentiment != "Select..." and confidence != "Select...":
                    confidence_score = int(confidence[0])
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Save to Google Sheet
                    if save_label_to_sheet(
                            st.session_state.sheet,
                            st.session_state.message_index,
                            st.session_state.current_message,
                            sentiment,
                            confidence_score,
                            labeler_name,
                            timestamp
                    ):
                        st.success(
                            f"✅ Labeled as **{sentiment}** (Confidence: {confidence_score}/5) and saved to Google Sheets!")
                        st.session_state.session_labeled_ids.add(st.session_state.message_index)
                        st.balloons()

                        st.session_state.current_message = None
                        st.session_state.message_index = None
                    else:
                        st.error("❌ Failed to save to Google Sheets")

                else:
                    st.error("⚠️ Please select both sentiment and confidence!")
    else:
        if st.session_state.dataset_loaded and st.session_state.sheet_connected and labeler_name:
            st.info("Click 'Load Next Message' to continue labeling.")

    # progress tracker
    st.divider()
    st.markdown("### 📈 Labeling Progress (Persistent)")

    col1, col2, col3 = st.columns(3)
    progress_count = min(effective_completed, TARGET_PER_USER)
    with col1:
        st.metric("Messages Labeled", progress_count)
    with col2:
        st.metric("Target", TARGET_PER_USER, "per user")
    with col3:
        progress_pct = min((progress_count / TARGET_PER_USER) * 100, 100)
        st.metric("Progress", f"{progress_pct:.1f}%")

    progress_bar = st.progress(min(progress_count / TARGET_PER_USER, 1.0))
    st.caption(
        f"Sentiment targets: Negative {SENTIMENT_TARGETS['Negative']}, "
        f"Neutral {SENTIMENT_TARGETS['Neutral']}, Positive {SENTIMENT_TARGETS['Positive']}"
    )
    st.caption(
        f"Completed: Negative {overlap_counts['Negative']}, "
        f"Neutral {overlap_counts['Neutral']}, Positive {overlap_counts['Positive']}"
    )

    # show all data from sheet
    st.divider()
    st.markdown("### 💾 All Labels (From Google Sheets)")

    if st.button("🔄 Refresh All Labels", use_container_width=True):
        with st.spinner("Loading all labels from Google Sheets..."):
            df_all = load_labels_from_sheet(st.session_state.sheet)

            if not df_all.empty:
                st.success(f"✅ Loaded {len(df_all)} total labels")

                # stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Labels", len(df_all))
                with col2:
                    if labeler_name:
                        df_user = df_all[
                            df_all["labeled_by"].fillna("").astype(str).str.strip().str.lower() == labeler_name.lower()
                        ]
                        st.metric(f"Your Labels", len(df_user))
                with col3:
                    st.metric("Team Members", df_all['labeled_by'].nunique())

                st.subheader("Sentiment Distribution (All)")
                sentiment_counts = df_all['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)

                st.subheader("Labels by Team Member")
                labeler_counts = df_all['labeled_by'].value_counts()
                st.bar_chart(labeler_counts)

                st.subheader("Recent Labels (Latest 15)")
                st.dataframe(df_all.tail(50).iloc[::-1], use_container_width=True)
            else:
                st.info("No labels yet. Start labeling!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Twitch Sentiment Labeling | CS 175 Project<br>
Labels synced to Google Sheets
</div>
""", unsafe_allow_html=True)
