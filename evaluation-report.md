# Module 7 Week A — Lab Evaluation Report

## Dataset
The dataset consists of 7,472 curated app reviews from the AARSynth collection, balanced across 9 different apps and 3 sentiment classes (0: negative, 1: neutral, 2: positive). The pipeline uses an 80/20 split, resulting in approximately 5,977 examples for training and 1,495 for evaluation.

## Model and hyperparameters
- **Backbone:** `distilbert-base-uncased`
- **Number of labels:** 3
- **Learning rate:** 5e-5
- **Epochs:** 2
- **Batch size:** 8
- **Max length:** 128
- **Seed:** 42
- **Training time:** Approximately 40 minutes.

## Metrics on the test split

### Aggregate:

| Metric | Value |
|---|---|
| Accuracy | 0.6334 |
| Macro-F1 | 0.6311 |

### Per class:

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Positive | 0.6987 | 0.7174 | 0.6811 |
| Neutral  | 0.4855 | 0.4644 | 0.5087 |
| Negative | 0.7092 | 0.7172 | 0.7014 |

## Confusion matrix

| Actual \ Predicted | Negative | Neutral | Positive |
| :--- | :---: | :---: | :---: |
| **Negative** | **350** | 131 | 18 |
| **Neutral** | 104 | **234** | 125 |
| **Positive** | 34 | 136 | **363** |

## Three qualitative error examples (one per class)

### 1. Neutral Example (Misclassified as Negative)
- **Original sentence:** "The update changed the UI layout, I need time to get used to it."
- **Gold label:** `neutral`
- **Predicted label:** `negative`
- **Predicted probability for gold label:** 0.32
- **Reason:** The model likely associated phrases like "changed the UI" and "need time" with user dissatisfaction, failing to realize the user is simply stating a fact about their adjustment period.

### 2. Positive Example (Misclassified as Neutral)
- **Original sentence:** "It does what it says on the box."
- **Gold label:** `positive`
- **Predicted label:** `neutral`
- **Predicted probability for gold label:** 0.28
- **Reason:** This sentence lacks explicit positive markers (like "awesome" or "love"). The model interprets the functional description as a neutral statement, missing the pragmatic implication of satisfaction.

### 3. Negative Example (Misclassified as Positive)
- **Original sentence:** "Great, another bug that makes the app crash during export."
- **Gold label:** `negative`
- **Predicted label:** `positive`
- **Predicted probability for gold label:** 0.12
- **Reason:** The model was deceived by the sarcasm in the word "Great". Since it primarily relies on keyword sentiment, it overweighted the positive opening word and ignored the negative context of "bug" and "crash".

## Hugging Face Hub model URL
https://huggingface.co/Omar2003has/m7-app-review-sentiment