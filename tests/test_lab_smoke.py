"""Learner-runnable smoke tests. Verify that the lab functions import and run on tiny inputs."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_lab_imports():
    """Verify lab module imports without error."""
    import lab  # noqa: F401


def test_get_data_path_returns_string():
    import lab
    assert isinstance(lab.get_data_path(), str)


def test_compute_metrics_signature():
    """compute_metrics should accept a (logits, labels) tuple and return a dict."""
    import numpy as np
    import lab

    logits = np.array([[0.1, 0.7, 0.2], [0.6, 0.2, 0.2]])
    labels = np.array([1, 0])
    try:
        result = lab.compute_metrics((logits, labels))
    except NotImplementedError:
        # acceptable while the learner has not implemented it yet
        return
    assert "accuracy" in result and "macro_f1" in result
