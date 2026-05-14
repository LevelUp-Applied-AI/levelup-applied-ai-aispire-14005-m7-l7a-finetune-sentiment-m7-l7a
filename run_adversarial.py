"""
Stretch Thursday — Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.
    """
    df = pd.read_csv(adv_csv_path)
    
    predicted_labels = []
    predicted_probs = []
    correct_list = []

    model.eval()
    id2label = model.config.id2label

    for _, row in df.iterrows():
        inputs = tokenizer(row['text'], return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
            prob, idx = torch.max(probs, dim=1)
            pred_idx = idx.item()
            pred_label = id2label[pred_idx]
            
            predicted_labels.append(pred_label)
            predicted_probs.append(prob.item())
            
            is_correct = (pred_label.lower() == str(row['expected_label']).lower())
            correct_list.append(is_correct)

    df['predicted_label'] = predicted_labels
    df['predicted_probability'] = predicted_probs
    df['correct'] = correct_list
    
    return df


def main() -> None:
    """Orchestrate; write results.csv."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")


if __name__ == "__main__":
    main()