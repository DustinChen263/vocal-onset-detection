"""Plotting helpers for qualitative onset-detection analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_onsets_on_spectrogram(audio: np.ndarray,
                               sr: int,
                               ref_onsets: Optional[np.ndarray] = None,
                               est_onsets: Optional[np.ndarray] = None,
                               tolerance: float = 0.05,
                               title: str = "",
                               t_start: float = 0.0,
                               t_end: Optional[float] = None,
                               ax: Optional[plt.Axes] = None,
                               figsize: tuple = (14, 4)) -> plt.Axes:
    """Show a log-mel spectrogram with reference (green) and estimated
    (red dashed) onsets overlaid.

    Onsets that are matched within ``tolerance`` are drawn solid; unmatched
    references (missed) are dotted green; unmatched estimates (false
    positives) are dashed red.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if t_end is None:
        t_end = len(audio) / sr
    s_start, s_end = int(t_start * sr), int(t_end * sr)
    seg = audio[s_start:s_end]

    S = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=2048,
                                       hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis="time", y_axis="mel", ax=ax)

    matched_ref, matched_est, missed, fps = _match_onsets(
        ref_onsets, est_onsets, tolerance)

    # offset times so they line up with the cropped audio (x-axis starts at 0)
    def _shift(arr):
        if arr is None or len(arr) == 0:
            return np.array([])
        a = np.asarray(arr) - t_start
        return a[(a >= 0) & (a <= t_end - t_start)]

    matched_ref = _shift(matched_ref)
    matched_est = _shift(matched_est)
    missed      = _shift(missed)
    fps         = _shift(fps)

    for t in matched_ref:
        ax.axvline(t, color="lime", linewidth=2.0, alpha=0.9)
    for t in missed:
        ax.axvline(t, color="lime", linewidth=2.0, alpha=0.9, linestyle=":")
    for t in matched_est:
        ax.axvline(t, color="red", linewidth=1.0, alpha=0.7, linestyle="--")
    for t in fps:
        ax.axvline(t, color="red", linewidth=1.5, alpha=0.9, linestyle="--")

    legend = [
        plt.Line2D([0], [0], color="lime", lw=2, label="ref (matched)"),
        plt.Line2D([0], [0], color="lime", lw=2, ls=":",
                   label="ref (missed = FN)"),
        plt.Line2D([0], [0], color="red",  lw=1, ls="--",
                   label="est (matched)"),
        plt.Line2D([0], [0], color="red",  lw=1.5, ls="--",
                   label="est (FP)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_title(title)
    return ax


def _match_onsets(ref: Optional[np.ndarray],
                  est: Optional[np.ndarray],
                  tol: float) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """Greedy bipartite match within ``tol``. Returns
    (matched_ref, matched_est, missed_ref, false_positive_est)."""
    ref = np.sort(ref) if ref is not None else np.array([])
    est = np.sort(est) if est is not None else np.array([])

    used_est = np.zeros(len(est), dtype=bool)
    matched_ref, matched_est, missed = [], [], []

    for r in ref:
        if len(est) == 0:
            missed.append(r); continue
        diffs = np.abs(est - r)
        diffs[used_est] = np.inf
        i = int(np.argmin(diffs))
        if diffs[i] <= tol:
            matched_ref.append(r); matched_est.append(est[i])
            used_est[i] = True
        else:
            missed.append(r)

    fps = est[~used_est]
    return (np.asarray(matched_ref), np.asarray(matched_est),
            np.asarray(missed), np.asarray(fps))


def plot_three_methods(track, audio: np.ndarray, sr: int,
                       estimates: dict[str, np.ndarray],
                       tolerance: float = 0.05,
                       t_start: float = 0.0,
                       t_end: Optional[float] = None,
                       figsize: tuple = (14, 9)) -> plt.Figure:
    """Stack three spectrogram plots (one per detector) for the same track."""
    fig, axes = plt.subplots(len(estimates), 1, figsize=figsize, sharex=True)
    if len(estimates) == 1:
        axes = [axes]
    for ax, (name, est) in zip(axes, estimates.items()):
        plot_onsets_on_spectrogram(
            audio, sr,
            ref_onsets=track.onsets, est_onsets=est,
            tolerance=tolerance, title=f"{track.track_id} — {name}",
            t_start=t_start, t_end=t_end, ax=ax,
        )
    fig.tight_layout()
    return fig


def categorize_errors(ref: np.ndarray, est: np.ndarray,
                      tol: float = 0.05,
                      doubletrig_tol: float = 0.12) -> dict:
    """Quick error taxonomy. For each FP, check if it is:
        - 'double_trigger'  : another est within ``doubletrig_tol`` of an FP
                              that *did* match a ref (suggests two estimates
                              landed near the same true onset)
        - 'isolated_fp'     : everything else
    For each FN, check whether the nearest est was within ``2*tol`` (i.e.,
    'shifted' — algorithm fired close but missed the window).
    """
    matched_ref, matched_est, missed, fps = _match_onsets(ref, est, tol)

    double_trigger = 0
    isolated_fp = 0
    for fp in fps:
        if len(matched_est) and np.min(np.abs(matched_est - fp)) < doubletrig_tol:
            double_trigger += 1
        else:
            isolated_fp += 1

    shifted = 0
    hard_miss = 0
    for m in missed:
        if len(est) and np.min(np.abs(est - m)) < 2 * tol:
            shifted += 1
        else:
            hard_miss += 1

    return {
        "tp":             len(matched_ref),
        "fp_double":      double_trigger,
        "fp_isolated":    isolated_fp,
        "fn_shifted":     shifted,
        "fn_hardmiss":    hard_miss,
    }
