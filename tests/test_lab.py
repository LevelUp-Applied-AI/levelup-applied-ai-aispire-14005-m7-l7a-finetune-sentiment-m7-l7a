"""
Lab 7A autograder.

Runs against the learner's repo root after `python lab.py` has executed
end-to-end on the smoke fixture (see workflow YAML).
"""
import ast
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------------------
# Drill-style mechanical checks (functions + signatures)
# ---------------------------------------------------------------------------

def test_prepare_dataset_returns_dict_with_train_and_test():
    import lab
    ds = lab.prepare_dataset(os.path.join(REPO_ROOT, "fixtures", "tiny_app_reviews.csv"), test_size=0.2, seed=42)
    assert "train" in ds and "test" in ds


def test_prepare_dataset_split_sizes():
    import lab
    ds = lab.prepare_dataset(os.path.join(REPO_ROOT, "fixtures", "tiny_app_reviews.csv"), test_size=0.2, seed=42)
    total = len(ds["train"]) + len(ds["test"])
    assert abs(len(ds["test"]) - total * 0.2) <= 2


def test_tokenize_dataset_columns():
    import lab
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = lab.prepare_dataset(os.path.join(REPO_ROOT, "fixtures", "tiny_app_reviews.csv"), test_size=0.2, seed=42)
    tokenized = lab.tokenize_dataset(ds, tokenizer, max_length=64)
    for split in ("train", "test"):
        cols = tokenized[split].column_names
        assert "input_ids" in cols and "attention_mask" in cols


def test_tokenize_dataset_max_length_truncates():
    import lab
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = lab.prepare_dataset(os.path.join(REPO_ROOT, "fixtures", "tiny_app_reviews.csv"), test_size=0.2, seed=42)
    tokenized = lab.tokenize_dataset(ds, tokenizer, max_length=16)
    max_seen = max(len(x) for x in tokenized["train"]["input_ids"])
    assert max_seen <= 16


def test_make_training_args_attributes():
    import lab
    args = lab.make_training_args("model", lr=3e-5, epochs=1, batch_size=4, seed=7)
    assert args.learning_rate == 3e-5
    assert args.num_train_epochs == 1
    assert args.per_device_train_batch_size == 4
    assert args.seed == 7
    # Per the lab guide, set evaluation/save cadence to once per epoch and
    # logging cadence to every ~50 steps. The eval_strategy attribute is named
    # eval_strategy (not evaluation_strategy) in transformers>=4.41 — the
    # course pins that range in requirements.txt.
    # Use equality (not `str(...) == "epoch"`): `eval_strategy` and
    # `save_strategy` are `IntervalStrategy` enum members whose `str()` output
    # changed in transformers 4.57.x (returns "IntervalStrategy.EPOCH",
    # not "epoch"). Equality works correctly because IntervalStrategy is a
    # str subclass; the enum compares against the underlying value regardless
    # of __str__ behavior. See pilot-learnings/module-7-week-a-eval-strategy-test-bug.md.
    assert args.eval_strategy == "epoch", \
        f"eval_strategy must be 'epoch' (got {args.eval_strategy!r})"
    assert args.save_strategy == "epoch", \
        f"save_strategy must be 'epoch' (got {args.save_strategy!r})"
    assert args.logging_steps == 50, \
        f"logging_steps must be 50 (got {args.logging_steps})"


def test_compute_metrics_shape():
    import lab
    logits = np.array([[0.1, 0.7, 0.2], [0.6, 0.2, 0.2]])
    labels = np.array([1, 0])
    result = lab.compute_metrics((logits, labels))
    assert "accuracy" in result and "macro_f1" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0


def test_compute_metrics_perfect_predictions():
    import lab
    logits = np.array([[0.1, 0.9, 0.0], [0.9, 0.05, 0.05], [0.0, 0.1, 0.9]])
    labels = np.array([1, 0, 2])
    result = lab.compute_metrics((logits, labels))
    assert abs(result["accuracy"] - 1.0) < 1e-9
    assert abs(result["macro_f1"] - 1.0) < 1e-9


def test_compute_metrics_known_confusion():
    """Hand-computed: 4 samples, 2 correct → accuracy 0.5; per-class F1 known."""
    import lab
    # true: [0, 0, 1, 1]; preds: [0, 1, 0, 1]
    # class 0: P=1/2, R=1/2, F1=0.5; class 1: P=1/2, R=1/2, F1=0.5; macro=0.5
    logits = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    result = lab.compute_metrics((logits, labels))
    assert abs(result["accuracy"] - 0.5) < 1e-9
    assert abs(result["macro_f1"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# End-to-end pipeline checks (CI runs python lab.py first)
# ---------------------------------------------------------------------------

def test_metrics_json_exists_and_well_formed():
    path = os.path.join(REPO_ROOT, "metrics.json")
    assert os.path.isfile(path), "metrics.json must be produced by lab.py"
    with open(path) as f:
        m = json.load(f)
    for key in ("accuracy", "macro_f1", "per_class_f1", "per_class_precision", "per_class_recall"):
        assert key in m, f"metrics.json must include '{key}' (Outcome 5: accuracy, precision, recall, macro-F1, confusion matrix)"
    # Per-class dicts must use the same string label keys (read from id2label)
    f1_labels = set(m["per_class_f1"].keys())
    assert set(m["per_class_precision"].keys()) == f1_labels, \
        "per_class_precision must use the same label keys as per_class_f1"
    assert set(m["per_class_recall"].keys()) == f1_labels, \
        "per_class_recall must use the same label keys as per_class_f1"
    # Per-class values are in [0, 1]
    for d_name in ("per_class_f1", "per_class_precision", "per_class_recall"):
        for label, v in m[d_name].items():
            assert 0.0 <= v <= 1.0, f"{d_name}[{label}] = {v} out of [0,1]"


def test_train_classifier_smoke_loss_decreased():
    """Liveness check on the smoke fixture training run — replaces the prior
    accuracy-threshold gate.

    A 60-row fixture × 0.2 test split = 12 test rows. On a 3-class problem,
    each prediction flip swings accuracy by ~8.3 pts. Even a 0.4 floor flaked
    on correctly-wired training (observed run: accuracy=0.167 with the model
    collapsed to one class). Loss is a strictly stronger signal — it shows
    whether training *actually learned*, independent of small-fixture
    stochasticity.

    main() persists trainer.state.log_history to training_log.json. The
    Trainer's final summary entry contains `train_loss` (the mean training
    loss over the full run). For 3-class random initialization, loss starts
    near -ln(1/3) ≈ 1.099. A correctly-wired DistilBERT fine-tune over 2
    epochs reliably drives mean train_loss below 1.0 even on the tiny smoke
    fixture (verified during build). Loss above 1.0 means training did not
    learn — a signal of broken wiring (optimizer not stepping, labels
    misaligned, train_dataset not passed to Trainer, etc.).
    """
    path = os.path.join(REPO_ROOT, "training_log.json")
    assert os.path.isfile(path), (
        "training_log.json must be produced by lab.py main() — "
        "json.dump(trainer.state.log_history, f) after trainer.save_model"
    )
    with open(path) as f:
        log = json.load(f)
    final = next((e for e in log if "train_loss" in e), None)
    assert final is not None, (
        "log_history has no final train_loss entry — Trainer.train() did not "
        "complete. Check that train_dataset is passed to Trainer."
    )
    assert final["train_loss"] < 1.0, (
        f"Mean train_loss = {final['train_loss']:.4f} after 2 epochs; 3-class "
        f"random init sits at ~1.099, so loss above 1.0 means training did "
        f"not learn meaningfully. Common causes: optimizer not stepping, "
        f"learning rate misconfigured, labels not aligned with id2label, "
        f"or train_dataset not actually passed to Trainer."
    )


def test_predictions_csv_has_required_columns():
    """predictions.csv carries per-row labels + predicted-class probability +
    the full per-class probability distribution (one prob_<label> column per
    class). The per-class columns enable the evaluation-report's gold-class
    probability ask on misclassifications and feed the Tuesday Honors
    calibration stretch.
    """
    path = os.path.join(REPO_ROOT, "predictions.csv")
    assert os.path.isfile(path), "predictions.csv must be produced by lab.py"
    df = pd.read_csv(path)
    for col in ["text", "label", "predicted_label", "predicted_probability"]:
        assert col in df.columns, f"missing column: {col}"
    assert df["predicted_probability"].between(0.0, 1.0).all()
    # Per-class probability columns: at least one prob_<label>, all in [0,1],
    # and rows must sum to ~1 (catches a learner who wrote raw logits instead
    # of softmax probabilities).
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    assert len(prob_cols) >= 2, (
        f"predictions.csv must include per-class probability columns named "
        f"prob_<label> (one per class); found: {prob_cols}"
    )
    for c in prob_cols:
        assert df[c].between(0.0, 1.0).all(), \
            f"{c} must be in [0,1] — you may be writing logits instead of softmax probabilities"
    row_sums = df[prob_cols].sum(axis=1)
    assert row_sums.between(0.99, 1.01).all(), (
        f"per-class probability columns must sum to ~1 per row (softmax); "
        f"observed sums in [{row_sums.min():.3f}, {row_sums.max():.3f}]. "
        f"You may be writing raw logits instead of probabilities."
    )


def test_confusion_matrix_csv_persisted():
    """Outcome 5 names the confusion matrix as required evidence — it must be
    a committed artifact, not just stdout. main() writes it as a square CSV
    indexed by true_label with columns = predicted_label class names.
    """
    path = os.path.join(REPO_ROOT, "confusion_matrix.csv")
    assert os.path.isfile(path), "confusion_matrix.csv must be produced by lab.py main()"
    df = pd.read_csv(path, index_col=0)
    # Square matrix
    assert df.shape[0] == df.shape[1], \
        f"confusion_matrix.csv must be square (got {df.shape})"
    assert df.shape[0] >= 2, "confusion matrix must have at least 2 classes"
    # Rows and columns share the same label set
    assert set(df.index) == set(df.columns), \
        "row labels (true) must match column labels (predicted)"
    # All cells are non-negative integer counts
    assert (df.values >= 0).all(), "confusion-matrix cells must be non-negative"


def test_train_classifier_produces_local_checkpoint():
    """After main() runs in CI, model/ dir must exist with weights + config."""
    model_dir = os.path.join(REPO_ROOT, "model")
    assert os.path.isdir(model_dir), "model/ should exist locally after training"
    files = os.listdir(model_dir)
    weights = any(f in {"pytorch_model.bin", "model.safetensors"} for f in files)
    assert weights, "model/ must contain pytorch_model.bin or model.safetensors"
    assert "config.json" in files


def test_main_produces_committed_artifacts():
    """metrics.json + predictions.csv must exist (committed); model/ exists locally but not tracked."""
    assert os.path.isfile(os.path.join(REPO_ROOT, "metrics.json"))
    assert os.path.isfile(os.path.join(REPO_ROOT, "predictions.csv"))


# ---------------------------------------------------------------------------
# Gitignore + tracking checks
# ---------------------------------------------------------------------------

def test_model_directory_gitignored():
    """model/ must be in .gitignore AND not tracked by git."""
    gitignore = os.path.join(REPO_ROOT, ".gitignore")
    assert os.path.isfile(gitignore)
    with open(gitignore) as f:
        contents = f.read()
    assert "model/" in contents or "model" in contents.split("\n"), \
        ".gitignore must include a `model/` entry"

    try:
        ls = subprocess.run(
            ["git", "ls-files", "model/"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert not ls.stdout.strip(), \
            f"model/ must not be tracked. Tracked files: {ls.stdout!r}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("git not available in CI environment")
