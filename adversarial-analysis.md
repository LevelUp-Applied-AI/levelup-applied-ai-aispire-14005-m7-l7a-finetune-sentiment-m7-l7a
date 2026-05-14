
# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis category  | Correct | Total | Accuracy 
| negation             | 3       | 7     | 42.8% 
| lexical_trigger      | 3       | 7     | 42.8% 
| domain_shift         | 3       | 6     | 50.0% 
| length_extreme       | 4       | 5     | 80.0% 
| sarcasm              | 1       | 5     | 20.0% 
| other                | 1       | 3     | 33.3% 

## Confirmed hypotheses

* **Sarcasm (IDs 16, 17, 18, 23):** The model completely failed to detect sarcasm as predicted. For instance, in **ID 17**, it predicted "positive" with 73% probability for the sentence "I really love waiting ten minutes...", proving it prioritizes the word "love" over the negative context of long wait times.
* **Double Negation (IDs 5, 6):** The results confirmed the model's struggle with complex syntactic negation. In **ID 5** ("can't say I'm not disappointed"), the model failed to resolve the double negative and incorrectly predicted "neutral" instead of "negative."
* **Lexical Triggers (IDs 9, 20, 28):** Strong sentiment "cues" easily misled the model. In **ID 9**, the presence of the word "perfect" triggered a 75% positive prediction, despite the sentence explicitly describing a "disaster."

## Refuted hypotheses

* **Length Extreme (IDs 13, 15):** The model handled extremely short inputs much better than expected. It correctly identified the sentiment for "Fix it" (negative) and "Wow" (positive) with high confidence, showing robustness to minimal context.
* **Basic Negation (IDs 1, 4, 19):** I hypothesized that the model might miss simple negation cues, but it successfully handled sentiment flipping in sentences like "did not improve" or "not bad," suggesting it has learned basic linguistic reversals.

## What the results reveal about the decision boundary

The adversarial results reveal that the model’s decision boundary is heavily biased toward **lexical valence (individual word weights)** rather than **global syntactic context**. If a sentence contains a high-polarity word like "love" or "perfect," the model's prediction shifts toward that polarity regardless of surrounding negations or sarcastic intent. Additionally, the model shows instability in its "neutral" boundary when facing **Domain Shift** (sports or finance), often forcing non-sentimental facts into "negative" or "neutral" categories with low confidence, indicating it is over-optimized for finding sentiment even where none exists.

