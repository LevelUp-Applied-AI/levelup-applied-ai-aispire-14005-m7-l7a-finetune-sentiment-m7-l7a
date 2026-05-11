# Module 7 Week A — Lab Evaluation Report

## Dataset

AARSynth app reviews (Sentences-50Agree) — 7,472 labeled reviews across 9 apps (Bitmoji, AccuWeather, Adobe Acrobat Reader, Adobe Lightroom, Booking.com, Forest, Slack, UC Browser, BBM), with 3 sentiment classes: 0=negative, 1=neutral, 2=positive. An 80/20 internal split (seed=42) yields 5,977 training examples and 1,495 test examples.

## Model and hyperparameters

- **Backbone:** distilbert-base-uncased
- **Number of labels:** 3 (negative, neutral, positive)
- **Learning rate:** 5e-5
- **Epochs:** 2
- **Batch size:** 8
- **max_length:** 128
- **Seed:** 42
- **Training time (wall-clock):** ~13 minutes on CPU (AMD Ryzen 7 5800H, no discrete GPU)

## Metrics on the test split

Aggregate:

| Metric   | Value  |
|----------|--------|
| Accuracy | 0.6334 |
| Macro-F1 | 0.6315 |

Per class (read from `metrics.json`):

| Class    | F1     | Precision | Recall |
|----------|--------|-----------|--------|
| Negative | 0.7124 | 0.7175    | 0.7074 |
| Neutral  | 0.4850 | 0.4644    | 0.5076 |
| Positive | 0.6971 | 0.7223    | 0.6735 |



## Confusion matrix

| true\pred | negative | neutral | positive |
|-----------|----------|---------|----------|
| negative  | 353      | 128     | 18       |
| neutral   | 108      | 235     | 120      |
| positive  | 31       | 143     | 359      |

The neutral class is hardest to identify — the model most frequently confuses neutral↔positive (120+143 errors), likely because many neutral reviews contain positive-sounding language without strong sentiment.

## Three qualitative error examples (one per class)

### Error 1 — Misclassified Negative

- **Text:** "can we get this to, somehow, plant an actual tree? say, for a certain amount of trees grown successfully a tree gets planted at some forest in need."
- **Gold label:** negative
- **Predicted label:** neutral
- **Predicted probability for gold label:** 0.061
- **Analysis:** This review is a feature request phrased as a hopeful suggestion, with no explicitly negative words like "bad", "crash", or "hate". The model likely treated the constructive, polite tone as neutral rather than reading the implicit dissatisfaction — the user wants something the app doesn't currently do.

### Error 2 — Misclassified Neutral

- **Text:** "nice app to use with friends"
- **Gold label:** neutral
- **Predicted label:** positive
- **Predicted probability for gold label:** 0.063
- **Analysis:** The word "nice" is a strong positive signal that consistently pushes the model toward the positive class. The 3-star neutral label here reflects a lukewarm rating rather than negative feedback, but the surface wording is indistinguishable from a genuine positive — this is a hard case even for a human annotator.

### Error 3 — Misclassified Positive

- **Text:** "good, but slow workflow."
- **Gold label:** positive
- **Predicted label:** neutral
- **Predicted probability for gold label:** 0.202
- **Analysis:** Mixed-sentiment reviews are a known challenge. "Good" signals positive, but "slow workflow" introduces a concrete complaint that pulls the prediction toward neutral. The model correctly senses the mixed signal but misses the net-positive label — negative qualifiers in the second clause tend to dominate attention.

## Hugging Face Hub model URL

https://huggingface.co/m02rashdan/m7-app-review-sentiment