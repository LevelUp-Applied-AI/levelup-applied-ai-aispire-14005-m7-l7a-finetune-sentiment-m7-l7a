# Data

The lab's training dataset is committed to this directory:

- **`app_reviews_train.csv`** — 7,472 reviews curated from <a href="https://huggingface.co/datasets/recmeapp/AARSynth" target="_blank">recmeapp/AARSynth</a> across 9 recognizable apps (Bitmoji, AccuWeather, Adobe Acrobat Reader, Adobe Lightroom, Booking.com, Forest, Slack, UC Browser, BBM), with star ratings mapped to 3 sentiment classes (1–2 → 0 negative; 3 → 1 neutral; 4–5 → 2 positive). Balanced across (app, class) buckets. The lab pipeline loads this file by default; `prepare_dataset` does an 80/20 internal split (seed=42) → ~5,977 train / ~1,495 eval.
- **`app_reviews_eval.csv`** — 1,867 reviews from the same curation pipeline, set aside as an instructor-side holdout. Not consumed by the default lab flow; available for sanity-checking model performance on a clean held-out set.
- **`fixtures/tiny_app_reviews.csv`** — 60-row CI smoke fixture (used when `DATA_PATH=fixtures/tiny_app_reviews.csv` is set in the environment).

The curation script lives in the curriculum repo at `scripts/curation/curate_m7_aarsynth.py`. Re-running it produces the same CSVs (deterministic given a fixed random seed and input dataset version).
