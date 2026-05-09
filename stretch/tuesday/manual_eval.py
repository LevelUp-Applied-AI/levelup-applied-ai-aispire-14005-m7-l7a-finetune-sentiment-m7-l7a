"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    # TODO: iterate texts in batches
    # TODO: tokenize each batch with truncation, max_length=128, padding=True, return_tensors='pt'
    # TODO: forward pass under torch.no_grad()
    # TODO: softmax over the last dim
    # TODO: argmax to get class indices
    # TODO: collect into numpy arrays of shape (N,) and (N, num_classes); return both
    raise NotImplementedError


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.

    Returns:
      {
        "accuracy": float,
        "macro_f1": float,
        "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
      }
    """
    # TODO: compute true positives / false positives / false negatives per class
    # TODO: precision = TP / (TP + FP); guard divide-by-zero
    # TODO: recall = TP / (TP + FN)
    # TODO: f1 = 2 * P * R / (P + R)
    # TODO: accuracy = sum(y_pred == y_true) / N
    # TODO: macro-F1 = mean of per-class f1 scores
    # TODO: assemble and return the dict
    raise NotImplementedError
