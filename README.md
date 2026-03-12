# Twitch Sentiment

Twitch Sentiment is an end-to-end pipeline for labeling Twitch chat, training sentiment classifiers, and running live sentiment inference on a Twitch channel. It focuses on three sentiment classes: Negative, Neutral, and Positive.

The repository includes a labeled Twitch chat dataset, a Streamlit annotation tool, a feature-based logistic regression baseline, transformer fine-tuning scripts for DistilBERT and Cardiff Twitter RoBERTa, a live Twitch IRC listener, and a FastAPI + WebSocket browser dashboard for real-time sentiment display.

## Demo and Links

- Demo video: [Google Drive](https://drive.google.com/file/d/10jB-lijx7WXz_6mXF938G1N8oUdY7SGL/view)
- Hugging Face model: [`JDMates/TwitchRoBERTaSentiment`](https://huggingface.co/JDMates/TwitchRoBERTaSentiment)
- Notebook demo: `project_sentiment.ipynb`

## Project Layout
```text
.
|-- sentiment_datasets/
|   |-- Twitch_Sentiment_Labels.csv
|   `-- twitch_emote_vader_lexicon.txt
|-- src/
|   |-- advanced_lr_model.py
|   |-- bert_sentiment_model.py
|   |-- cardiff_sentiment_model.py
|   |-- labeler_app.py
|   `-- scrape_ffz.py
|-- web/
|   |-- chat.html
|   |-- chat.css
|   `-- chat.js
|-- display_bert.py
|-- twitch_listener.py
|-- web_cardiff_chat.py
`-- project_sentiment.ipynb
```

## What Each Part Does

### Dataset

`sentiment_datasets/Twitch_Sentiment_Labels.csv` stores message-level annotations with the following columns: `message_id`, `message`, `sentiment`, `confidence`, `labeled_by`, and `timestamp`. Training scripts collapse duplicate labels into a gold label using confidence-weighted voting. The `sentiment_datasets/` directory is included intentionally so the final labeled data used in the project is easy to inspect and reuse.

### Classical Baseline

`src/advanced_lr_model.py` trains a logistic regression classifier using word-level TF-IDF features, character n-grams, handcrafted message style features, and Twitch emote sentiment scores derived from our custom lexicon.

### Transformer Models

`src/bert_sentiment_model.py` fine-tunes a Hugging Face sequence classifier on the gold-labeled dataset. `src/cardiff_sentiment_model.py` is a thin wrapper around the same trainer, preconfigured for the CardiffNLP Twitter RoBERTa model. Both pipelines support emote-tag text augmentation, train/validation/test splits, optional Twitter-style normalization, and saved model artifacts under `data/`. The main model used by the live demo and notebook is the Cardiff-based checkpoint on Hugging Face: [`JDMates/TwitchRoBERTaSentiment`](https://huggingface.co/JDMates/TwitchRoBERTaSentiment).

### Live Inference

`display_bert.py` connects to Twitch IRC, loads a transformer sentiment model, and prints live classified messages in the terminal. `web_cardiff_chat.py` serves a browser dashboard that streams live chat and rolling sentiment statistics over WebSockets.

### Labeling Tool

`src/labeler_app.py` is a Streamlit app for assigning sentiment labels with optional Google Sheets sync, which is how all three team members labeled data collaboratively during the project.

## Setup

### 1. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Optional extras for the labeling tool and FFZ scraper:
```bash
pip install gspread oauth2client beautifulsoup4
```

### 3. Create `.env`

The live Twitch tools expect a local `.env` file. Never commit real secrets to version control.
```env
TWITCH_BOT_TOKEN=oauth:your_twitch_bot_token
TWITCH_BOT_NICK=YourBotName

# Live model loading expects a Hugging Face repo ID
MODEL_ID=your-huggingface-username/your-model-repo

# Optional
# MODEL_REVISION=
# HF_TOKEN=
# BERT_MAX_LENGTH=128
# BERT_MIN_CONFIDENCE=0.55
# BERT_MIN_MARGIN=0.10
```

To run the notebook and live demo without modifying any code, set:
```env
MODEL_ID=JDMates/TwitchRoBERTaSentiment
```

## Training

The dataset lives under `sentiment_datasets/`, so pass the path explicitly when running training scripts.

### Logistic Regression
```bash
python src/advanced_lr_model.py --data sentiment_datasets/Twitch_Sentiment_Labels.csv --lexicon sentiment_datasets/twitch_emote_vader_lexicon.txt --model_out data/lr_sentiment_model.joblib
```

### DistilBERT
```bash
python src/bert_sentiment_model.py --data sentiment_datasets/Twitch_Sentiment_Labels.csv --emote_lexicon sentiment_datasets/twitch_emote_vader_lexicon.txt --output_dir data/bert_sentiment_model
```

### Cardiff Twitter RoBERTa
```bash
python src/cardiff_sentiment_model.py --data sentiment_datasets/Twitch_Sentiment_Labels.csv --emote_lexicon sentiment_datasets/twitch_emote_vader_lexicon.txt --normalize_twitter --output_dir data/cardiff_sentiment_model
```

All scripts save checkpoints and model artifacts under `data/`. The live dashboard loads models via `MODEL_ID`, so using a freshly trained local model for live inference requires either uploading it to Hugging Face or adapting the loader.

## Running the Live Analyzer

### Terminal mode
```bash
python display_bert.py
```

You will be prompted for a Twitch channel name. The script connects to Twitch IRC, classifies each incoming message, and prints the sentiment label, confidence score, username, and message text in real time, with periodic aggregate statistics.

### Web dashboard
```bash
python -m uvicorn web_cardiff_chat:app --reload
```

Then open `http://127.0.0.1:8000`. The dashboard supports start/stop channel listening from the browser, a live message stream with sentiment badges, rolling positive/neutral/negative percentages, and a popout view for chat monitoring.

## Labeling More Data
```bash
streamlit run src/labeler_app.py
```

Google Sheets sync requires a service-account credentials file added to Streamlit secrets. Without it, the local annotation flow still works as the primary labeling interface.

## Scraping Emotes
```bash
python src/scrape_ffz.py
```

This writes a JSON file of FrankerFaceZ emote names that can be used for lexicon construction or feature engineering experiments.

## Notebook

`project_sentiment.ipynb` is the submission notebook for the project demo. It takes a fast, focused path through the project rather than reproducing the full training pipeline: it loads the fine-tuned Cardiff RoBERTa model from Hugging Face, reads a small sample chat log from `test.txt`, runs sentiment predictions on a contiguous 25-message window, and displays predicted labels and confidence scores inline.

The notebook is designed to be lightweight and demonstrative. The full training and live inference workflows are available in `src/`, `display_bert.py`, and `web_cardiff_chat.py`.

A few notes:

- The first run may take longer if the Hugging Face model is not cached locally. Subsequent runs will be significantly faster.
- `src/` is included to satisfy the course requirement that individual scripts be available for inspection.
- `sentiment_datasets/` is included because those files were used to build the final labeled dataset and are part of the project deliverables.
- Secrets such as Twitch tokens and API keys belong in `.env` and should never be uploaded to any repository, including private ones.