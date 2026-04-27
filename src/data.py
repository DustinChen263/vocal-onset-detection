"""Dataset loaders for vocadito and Dagstuhl ChoirSet.

Both datasets are wrapped behind a uniform ``Track`` interface so that the
rest of the pipeline does not care which source the data came from.

Audio is **lazily loaded** via ``Track.load_audio()`` to keep memory low.

Usage
-----
>>> from src.data import load_vocadito
>>> tracks = load_vocadito()
>>> tr = tracks[0]
>>> tr.dataset, tr.track_id
('vocadito', '1')
>>> audio, sr = tr.load_audio()           # native sample rate
>>> audio, sr = tr.load_audio(sr=22050)    # resampled
>>> tr.onsets[:5]                          # ground truth onset times in seconds
array([0.661..., 1.010..., 1.317..., ...])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import mirdata


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_data_home(data_home: str) -> str:
    """If ``data_home`` is relative, resolve it relative to the project root
    (the parent of ``src/``). Lets notebooks in any subdirectory use the
    same default ``"data/<dataset>"`` paths without breaking."""
    p = Path(data_home)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


@dataclass
class Track:
    """Uniform container for one audio clip + its onset ground truth."""

    dataset: str                       # e.g. "vocadito" or "dagstuhl_choirset"
    track_id: str                      # dataset-internal id
    audio_path: str                    # absolute path to .wav
    onsets: np.ndarray                 # ground-truth onset times (seconds)
    metadata: dict = field(default_factory=dict)

    def load_audio(self, sr: Optional[int] = None) -> tuple[np.ndarray, int]:
        """Load mono audio. ``sr=None`` keeps the file's native sample rate."""
        audio, native_sr = librosa.load(self.audio_path, sr=sr, mono=True)
        return audio, native_sr

    def __repr__(self) -> str:
        return (f"Track(dataset={self.dataset!r}, track_id={self.track_id!r}, "
                f"n_onsets={len(self.onsets)})")


# ---------------------------------------------------------------------------
# vocadito (solo vocals, 40 tracks)
# ---------------------------------------------------------------------------

def load_vocadito(data_home: str = "data/vocadito",
                  annotator: str = "a1") -> list[Track]:
    """Load all vocadito tracks.

    Parameters
    ----------
    data_home : str
        Path where mirdata downloaded vocadito (must already be downloaded).
    annotator : {"a1", "a2"}
        Which human annotator's notes to use as ground truth.
        vocadito ships two independent annotations; ``a1`` is the default.

    Returns
    -------
    list[Track]
    """
    if annotator not in ("a1", "a2"):
        raise ValueError(f"annotator must be 'a1' or 'a2', got {annotator!r}")

    voc = mirdata.initialize("vocadito", data_home=_resolve_data_home(data_home))
    tracks: list[Track] = []

    for tid in voc.track_ids:
        t = voc.track(tid)
        notes = getattr(t, f"notes_{annotator}")
        onsets = np.asarray(notes.intervals[:, 0], dtype=float)

        tracks.append(Track(
            dataset="vocadito",
            track_id=tid,
            audio_path=str(Path(t.audio_path).resolve()),
            onsets=onsets,
            metadata={
                "language":  t.language,
                "singer_id": t.singer_id,
                "annotator": annotator,
                "n_notes":   len(notes.intervals),
            },
        ))

    return tracks


# ---------------------------------------------------------------------------
# Dagstuhl ChoirSet (polyphonic SATB choir)
# ---------------------------------------------------------------------------

def load_dagstuhl(data_home: str = "data/dagstuhl_choirset",
                  mic: str = "stm",
                  dedup_tol: float = 0.03) -> list[Track]:
    """Load Dagstuhl ChoirSet multitracks that have onset annotations.

    Each multitrack with non-empty ``notes`` becomes one ``Track`` whose
    audio is the chosen room-mic recording (a polyphonic mix of the choir).
    Note onsets coming from different voices are de-duplicated within
    ``dedup_tol`` seconds (perceptually a single onset).

    Parameters
    ----------
    data_home : str
        Path where mirdata downloaded the dataset.
    mic : {"stm", "stl", "str", "rev"}
        Which room-mic audio to use:
        - "stm" : stereo middle (close mix) — recommended
        - "stl" / "str" : stereo left / right
        - "rev" : reverb mic (far)
    dedup_tol : float
        Onsets within this many seconds of each other are merged.
    """
    mic = mic.lower()
    if mic not in ("stm", "stl", "str", "rev"):
        raise ValueError(f"mic must be one of stm/stl/str/rev, got {mic!r}")

    dcs = mirdata.initialize("dagstuhl_choirset",
                             data_home=_resolve_data_home(data_home))
    tracks: list[Track] = []

    for mtid in dcs.mtrack_ids:
        mt = dcs.multitrack(mtid)
        if mt.notes is None:
            continue

        audio_path = getattr(mt, f"audio_{mic}_path")
        if audio_path is None:           # skip if this mic was not recorded
            continue

        all_onsets = np.asarray(mt.notes.intervals[:, 0], dtype=float)
        all_onsets.sort()
        merged = _merge_close(all_onsets, tol=dedup_tol)

        tracks.append(Track(
            dataset="dagstuhl_choirset",
            track_id=mtid,
            audio_path=str(Path(audio_path).resolve()),
            onsets=merged,
            metadata={
                "mic":            mic,
                "n_singers":      len(mt.track_ids),
                "n_score_events": len(all_onsets),
                "n_unique_onsets": len(merged),
                "piece":          mtid.split("_")[1],     # e.g. "LI"
                "ensemble":       mtid.split("_")[2],     # e.g. "FullChoir"
            },
        ))

    return tracks


def _merge_close(times: np.ndarray, tol: float) -> np.ndarray:
    """Greedy merge of sorted onset times within ``tol`` seconds."""
    if len(times) == 0:
        return times
    out = [times[0]]
    for t in times[1:]:
        if t - out[-1] > tol:
            out.append(t)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# CLI helper: ``python -m src.data --download``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download both datasets via mirdata")
    args = parser.parse_args()

    if args.download:
        print("[1/2] Downloading vocadito ...")
        mirdata.initialize("vocadito",
                           data_home="data/vocadito").download()
        print("[2/2] Downloading Dagstuhl ChoirSet ...")
        mirdata.initialize("dagstuhl_choirset",
                           data_home="data/dagstuhl_choirset").download()
        print("Done.")
    else:
        for loader, name in [(load_vocadito, "vocadito"),
                             (load_dagstuhl, "dagstuhl_choirset")]:
            tracks = loader()
            print(f"\n=== {name}: {len(tracks)} tracks ===")
            print("First:", tracks[0])
            audio, sr = tracks[0].load_audio()
            print(f"  audio:        shape={audio.shape}, sr={sr}")
            print(f"  first 5 onsets: {tracks[0].onsets[:5]}")
            print(f"  metadata:     {tracks[0].metadata}")
