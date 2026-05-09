"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    # TODO: bin edges via np.linspace(0, 1, n_bins + 1)
    # TODO: bucket_centers = midpoints of edges
    # TODO: for each prediction, take the max probability and the predicted class index
    # TODO: assign each prediction to a bucket by its max probability
    # TODO: bucket_accuracy = mean of (predicted == true) within the bucket; nan or 0 if empty
    # TODO: bucket_count = number of predictions in the bucket
    # TODO: return three numpy arrays
    raise NotImplementedError


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    # TODO: bucket predictions as in reliability_diagram
    # TODO: for each bucket, compute confidence (mean max probability) and accuracy
    # TODO: weight |accuracy - confidence| by bucket fraction; sum
    # TODO: return float
    raise NotImplementedError


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
