import argparse

from bert_sentiment_model import main as bert_main


def main():
    parser = argparse.ArgumentParser(
        description="Train a Cardiff Twitter-RoBERTa sentiment model on Twitch_Sentiment_Labels.csv",
        conflict_handler="resolve",
    )
    parser.add_argument("--data", default="Twitch_Sentiment_Labels.csv")
    parser.add_argument("--model_name", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_pct", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data/cardiff_sentiment_model")
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--normalize_twitter", action="store_true")
    parser.add_argument("--no_normalize_twitter", action="store_true")
    parser.add_argument("--emote_lexicon", default="twitch_emote_vader_lexicon.txt")
    parser.add_argument("--no_emote_tags", action="store_true", help="Disable emote-tag text augmentation")
    args = parser.parse_args()

    argv = [
        "--data",
        args.data,
        "--model_name",
        args.model_name,
        "--test_size",
        str(args.test_size),
        "--max_length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--train_batch_size",
        str(args.train_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--weight_decay",
        str(args.weight_decay),
        "--warmup_steps",
        str(args.warmup_steps),
        "--warmup_pct",
        str(args.warmup_pct),
        "--seed",
        str(args.seed),
        "--output_dir",
        args.output_dir,
        "--early_stopping_patience",
        str(args.early_stopping_patience),
        "--emote_lexicon",
        args.emote_lexicon,
    ]
    normalize_twitter = (True if not args.no_normalize_twitter else False) or bool(args.normalize_twitter)
    if normalize_twitter:
        argv.append("--normalize_twitter")
    if args.no_emote_tags:
        argv.append("--no_emote_tags")

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + argv
        bert_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
