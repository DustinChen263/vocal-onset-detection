"""Wrappers around mir_eval for onset evaluation.

To be filled in during Step 5.
"""

from __future__ import annotations
import numpy as np


def evaluate_track(ref_onsets: np.ndarray,
                   est_onsets: np.ndarray,
                   tolerance: float = 0.05) -> dict:
    """Compute precision, recall, F-measure for one track. TODO: Step 5."""
    raise NotImplementedError


def evaluate_dataset(predictions: dict, ground_truth: dict,
                     tolerance: float = 0.05):
    """Aggregate per-track metrics into a dataset-level table. TODO: Step 5."""
    raise NotImplementedError
