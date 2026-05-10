"""End-to-end source-separation experiment.

For every track in vocadito + Dagstuhl ChoirSet we evaluate three audio
*conditions* with the same three detectors and the same ground-truth
onsets, asking: does running a SOTA source separator on a vocal+
accompaniment mixture restore the onset-detection performance we get on
a clean vocal?

Conditions
----------
* ``clean``     — original audio (vocadito only — Dagstuhl has no clean stem)
* ``mix``       — clean vocal + MUSDB18 accompaniment at a controlled SNR
                  (vocadito only)
* ``mix_orig``  — the original Dagstuhl polyphonic room-mic mix (Dagstuhl only)
* ``separated`` — demucs(htdemucs) vocals stem of the corresponding ``mix``
                  or ``mix_orig`` input

The vocadito experiment is fully controlled: we know the clean reference,
we know the SNR, we know the demucs output. We sweep ``snr_db ∈ {-5, 0, +5}``.

The Dagstuhl experiment answers the more realistic question: "given a
real-world polyphonic vocal mixture (no isolated stem), does demucs help
even though it was *not* trained on choral data?"

Output
------
``results/separation_metrics.csv`` — long format, one row per
(dataset, track_id, condition, snr_db, detector, P, R, F).
"""

from __future__ import annotations

import src._env  # noqa: F401  — env shims must run before torch / madmom

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data import (
    Track,
    load_vocadito,
    load_dagstuhl,
    PROJECT_ROOT,
)
from src.detectors import DETECTORS
from src.evaluate import evaluate_track
from src.mixtures import (
    Accompaniment,
    SAMPLE_RATE,
    mix_at_snr,
    prepare_musdb_accompaniments,
)
from src.separation import separate_vocals


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SNRS = (-5.0, 0.0, 5.0)   # vocal-to-accompaniment SNR (dB)
TOLERANCE = 0.05                   # MIREX-standard ±50 ms


@dataclass
class _Variant:
    """One audio version of a track that we want to evaluate."""
    condition: str            # "clean", "mix", "mix_orig", "separated"
    snr_db: float | None      # only meaningful for vocadito mix/separated
    audio: np.ndarray         # mono, sr = SAMPLE_RATE
    sr: int


def _evaluate_variants(track: Track,
                       variants: list[_Variant],
                       acc_name: str | None) -> list[dict]:
    """Run all detectors on every variant; return one row per (variant, detector)."""
    rows: list[dict] = []
    for v in variants:
        for det_name, det_fn in DETECTORS.items():
            try:
                est = det_fn(v.audio, v.sr)
            except Exception as e:
                print(f"  [WARN] {det_name} failed on "
                      f"{track.dataset}/{track.track_id}/{v.condition}: {e}")
                est = np.array([])
            m = evaluate_track(track.onsets, est, tolerance=TOLERANCE)
            rows.append({
                "dataset":   track.dataset,
                "track_id":  track.track_id,
                "condition": v.condition,
                "snr_db":    v.snr_db,
                "accomp":    acc_name,
                "detector":  det_name,
                **m,
            })
    return rows


# ---------------------------------------------------------------------------
# vocadito sub-experiment (clean vs. mix vs. separated, multiple SNR)
# ---------------------------------------------------------------------------

def run_vocadito(snrs: Iterable[float] = DEFAULT_SNRS,
                 max_tracks: int | None = None,
                 seed: int = 0) -> pd.DataFrame:
    """For each vocadito track, mix at every SNR with one accompaniment
    chosen deterministically from the MUSDB18 sample, separate, and score
    every detector on (clean, mix, separated)."""

    vocadito = load_vocadito()
    if max_tracks is not None:
        vocadito = vocadito[:max_tracks]
    accs = prepare_musdb_accompaniments()
    rng = np.random.default_rng(seed)

    print(f"\n=== vocadito × {len(snrs)} SNRs ({list(snrs)} dB) "
          f"× {len(vocadito)} tracks ===")
    rows: list[dict] = []
    t0 = time.time()
    for i, tr in enumerate(vocadito, 1):
        clean, sr = tr.load_audio(sr=SAMPLE_RATE)
        # Pick one accompaniment per track. Deterministic for repeatability.
        acc = accs[rng.integers(0, len(accs))]
        acc_audio = acc.load(sr=SAMPLE_RATE)

        variants: list[_Variant] = [
            _Variant("clean", None, clean.astype(np.float32), SAMPLE_RATE),
        ]
        for snr in snrs:
            mix = mix_at_snr(clean, acc_audio, snr_db=snr,
                             rng=np.random.default_rng(seed + int(i)))
            variants.append(_Variant("mix", float(snr), mix, SAMPLE_RATE))
            sep, sr_out = separate_vocals(mix, SAMPLE_RATE)
            variants.append(_Variant("separated", float(snr), sep, sr_out))

        rows.extend(_evaluate_variants(tr, variants, acc_name=acc.name))

        elapsed = time.time() - t0
        eta = elapsed / i * (len(vocadito) - i)
        print(f"[{i:2d}/{len(vocadito)}] vocadito {tr.track_id:3s}  "
              f"acc={acc.name[:30]:30s}  "
              f"({elapsed:5.0f}s elapsed, ETA {eta:5.0f}s)")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dagstuhl sub-experiment (original mix vs. separated)
# ---------------------------------------------------------------------------

def run_dagstuhl(max_tracks: int | None = None) -> pd.DataFrame:
    """Run demucs on Dagstuhl's polyphonic room-mic mix and compare onset
    detection on the original mix vs. the separated vocal stem."""

    choir = load_dagstuhl()
    if max_tracks is not None:
        choir = choir[:max_tracks]

    print(f"\n=== Dagstuhl ChoirSet × {len(choir)} tracks "
          f"(mix_orig vs. separated) ===")
    rows: list[dict] = []
    t0 = time.time()
    for i, tr in enumerate(choir, 1):
        mix, sr = tr.load_audio(sr=SAMPLE_RATE)
        sep, sr_out = separate_vocals(mix, SAMPLE_RATE)
        variants = [
            _Variant("mix_orig",  None, mix.astype(np.float32), SAMPLE_RATE),
            _Variant("separated", None, sep,                    sr_out),
        ]
        rows.extend(_evaluate_variants(tr, variants, acc_name=None))

        elapsed = time.time() - t0
        eta = elapsed / i * (len(choir) - i)
        print(f"[{i:2d}/{len(choir)}] dagstuhl {tr.track_id:35s}  "
              f"({elapsed:5.0f}s elapsed, ETA {eta:5.0f}s)")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str,
                        default=str(PROJECT_ROOT / "results"
                                    / "separation_metrics.csv"))
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--max-vocadito", type=int, default=None,
                        help="Cap number of vocadito tracks (for quick runs)")
    parser.add_argument("--max-dagstuhl", type=int, default=None,
                        help="Cap number of Dagstuhl tracks")
    parser.add_argument("--skip-vocadito", action="store_true")
    parser.add_argument("--skip-dagstuhl", action="store_true")
    args = parser.parse_args()

    dfs = []
    if not args.skip_vocadito:
        dfs.append(run_vocadito(snrs=args.snrs, max_tracks=args.max_vocadito))
    if not args.skip_dagstuhl:
        dfs.append(run_dagstuhl(max_tracks=args.max_dagstuhl))

    df = pd.concat(dfs, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")

    print("\n=== Mean F-measure by (dataset, condition, snr_db, detector) ===")
    summary = (df.groupby(["dataset", "condition", "snr_db", "detector"],
                          dropna=False)
                 ["f_measure"]
                 .mean()
                 .round(3)
                 .unstack("detector"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
