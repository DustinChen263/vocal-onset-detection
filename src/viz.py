"""Plotting helpers used by the demo notebook (Step 6 / 7)."""

from __future__ import annotations
import numpy as np


def plot_onsets_on_spectrogram(audio: np.ndarray, sr: int,
                               ref_onsets: np.ndarray,
                               est_onsets: np.ndarray,
                               title: str = "") -> None:
    """Show a log-mel spectrogram with reference (green) and estimated
    (red dashed) onsets overlaid. TODO: Step 6."""
    raise NotImplementedError
