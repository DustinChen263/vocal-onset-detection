"""Onset-detection evaluation using mir_eval.

The standard MIR evaluation for onset detection (MIREX) is the F-measure
with a tolerance window of ``±50 ms``. We expose:

* :func:`evaluate_track`   — metrics for one track
* :func:`evaluate_all`     — run every detector on every track and return
                             a tidy ``pandas.DataFrame``

Usage
-----
>>> from src.data import load_vocadito
>>> from src.detectors import DETECTORS
>>> from src.evaluate import evaluate_all
>>> df = evaluate_all(load_vocadito(), DETECTORS)
>>> df.groupby('detector')[['precision', 'recall', 'f_measure']].mean()
"""

from __future__ import annotations

from typing import Callable, Iterable
import time

import numpy as np
import pandas as pd
import mir_eval

from src.data import Track


def evaluate_track(ref_onsets: np.ndarray,
                   est_onsets: np.ndarray,
                   tolerance: float = 0.05) -> dict:
    """Return precision / recall / F-measure for a single track.

    Parameters
    ----------
    ref_onsets, est_onsets : 1-D arrays of times in seconds (need not be sorted).
    tolerance : matching window in seconds. MIREX standard is 0.05.
    """
    ref = np.sort(np.asarray(ref_onsets, dtype=float))
    est = np.sort(np.asarray(est_onsets, dtype=float))

    f, p, r = mir_eval.onset.f_measure(ref, est, window=tolerance)
    return {
        "precision": p,
        "recall":    r,
        "f_measure": f,
        "n_ref":     len(ref),
        "n_est":     len(est),
    }


def evaluate_all(tracks: Iterable[Track],
                 detectors: dict[str, Callable[[np.ndarray, int], np.ndarray]],
                 tolerance: float = 0.05,
                 verbose: bool = True) -> pd.DataFrame:
    """Run every ``detector`` on every ``track`` and return a long-form DataFrame.

    Each row = one (track, detector) combination, with metrics + metadata.
    """
    rows = []
    tracks = list(tracks)
    total = len(tracks) * len(detectors)
    done = 0
    t0 = time.time()

    for tr in tracks:
        audio, sr = tr.load_audio()

        for det_name, det_fn in detectors.items():
            try:
                est = det_fn(audio, sr)
            except Exception as e:
                if verbose:
                    print(f"  [WARN] {det_name} failed on {tr.track_id}: {e}")
                est = np.array([])

            metrics = evaluate_track(tr.onsets, est, tolerance=tolerance)
            row = {
                "dataset":  tr.dataset,
                "track_id": tr.track_id,
                "detector": det_name,
                **metrics,
                **{f"meta_{k}": v for k, v in tr.metadata.items()},
            }
            rows.append(row)
            done += 1

            if verbose:
                print(f"[{done:3d}/{total}] {tr.dataset:20s} "
                      f"{tr.track_id:30s} {det_name:15s} "
                      f"F={metrics['f_measure']:.3f}")

    if verbose:
        print(f"\nFinished in {time.time() - t0:.1f}s")

    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (dataset, detector): mean F, P, R + std."""
    agg = (df.groupby(["dataset", "detector"])
             [["precision", "recall", "f_measure"]]
             .agg(["mean", "std"])
             .round(3))
    return agg


if __name__ == "__main__":
    from src.data import load_vocadito, load_dagstuhl
    from src.detectors import DETECTORS

    tracks = load_vocadito() + load_dagstuhl()
    print(f"Evaluating {len(tracks)} tracks × {len(DETECTORS)} detectors ...\n")

    df = evaluate_all(tracks, DETECTORS, tolerance=0.05)

    print("\n=== Per-dataset summary (mean ± std) ===")
    print(summary_table(df))

    out = "results/raw_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved per-track metrics to {out}")
