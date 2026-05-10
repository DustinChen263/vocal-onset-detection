"""Synthetic vocal+accompaniment mixtures for the source-separation experiment.

We take a vocadito *clean* solo vocal track (which has ground-truth onsets)
and add a polyphonic instrumental backing at a controlled signal-to-noise
ratio (SNR), where SNR is measured on the long-term RMS::

    SNR_dB = 20 * log10( rms(vocal) / rms(accompaniment_after_scaling) )

This gives us three audio variants per track for the same set of onsets:

* ``clean``     — the original vocadito vocal
* ``mix``       — vocal + scaled accompaniment
* ``separated`` — demucs-vocals(mix), produced separately by ``src.separation``

The accompaniment pool comes from the public **MUSDB18 sample** (7 free
tracks). We use the ``accompaniment`` stem (drums + bass + other), so it is
purely instrumental and does *not* contribute its own vocal onsets.
"""

from __future__ import annotations

import src._env  # noqa: F401  — sets KMP_DUPLICATE_LIB_OK / TORCH_HOME / NUMBA_CACHE_DIR

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

from src.data import PROJECT_ROOT


SAMPLE_RATE = 44100   # demucs htdemucs expects 44.1 kHz, stereo


# ---------------------------------------------------------------------------
# Accompaniment pool (MUSDB18 sample, 7 free tracks)
# ---------------------------------------------------------------------------

@dataclass
class Accompaniment:
    name: str
    audio_path: Path             # mono 44.1 kHz wav cached on disk

    def load(self, sr: int = SAMPLE_RATE) -> np.ndarray:
        a, _ = librosa.load(self.audio_path, sr=sr, mono=True)
        return a


def prepare_musdb_accompaniments(
    cache_dir: str | Path = "data/musdb_accompaniment",
    musdb_root: Optional[str | Path] = None,
    redownload: bool = False,
) -> list[Accompaniment]:
    """Download (if needed) the MUSDB18 *sample* and dump per-track
    instrumental backing as mono 44.1 kHz WAVs.

    The sample download happens via ``musdb.DB(download=True)`` which fetches
    a 7-track public subset (~30 MB) into ``musdb_root`` (defaults to
    ``data/musdb_sample/`` inside the project so we don't need write access
    outside the workspace).
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = sorted(cache_dir.glob("*.wav"))
    if cached and not redownload:
        return [Accompaniment(name=p.stem, audio_path=p) for p in cached]

    if musdb_root is None:
        musdb_root = PROJECT_ROOT / "data" / "musdb_sample"
    musdb_root = Path(musdb_root)

    import musdb  # lazy: only needed when populating the cache
    if not (musdb_root.exists() and any(musdb_root.iterdir())):
        # musdb.DB will mkdir(root) and download into it
        mus = musdb.DB(root=str(musdb_root), download=True)
    else:
        mus = musdb.DB(root=str(musdb_root))

    accs: list[Accompaniment] = []
    for tr in mus.tracks:
        # tr.targets['accompaniment'] = drums + bass + other (no vocals)
        acc_audio = tr.targets["accompaniment"].audio  # (n_samples, 2)
        mono = librosa.to_mono(acc_audio.T)
        if tr.rate != SAMPLE_RATE:
            mono = librosa.resample(mono, orig_sr=tr.rate, target_sr=SAMPLE_RATE)
        out = cache_dir / f"{tr.name}.wav"
        sf.write(out, mono, SAMPLE_RATE)
        accs.append(Accompaniment(name=tr.name, audio_path=out))

    return accs


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def mix_at_snr(vocal: np.ndarray,
               accompaniment: np.ndarray,
               snr_db: float,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Mix a mono vocal with mono accompaniment at the requested SNR.

    The accompaniment is looped or randomly cropped to match the vocal
    length, then scaled so that ``rms(vocal) / rms(accomp) = 10**(snr/20)``.
    The mixture is peak-normalized to ``0.99`` to avoid clipping.

    Parameters
    ----------
    vocal, accompaniment : 1-D float arrays at the same sample rate.
    snr_db : float
        Vocal-to-accompaniment ratio in dB.
        ``+inf`` returns the unmodified ``vocal``.
    rng : np.random.Generator, optional
        Used to randomly crop the accompaniment if it's longer than the
        vocal, and to pick a starting offset.
    """
    if not np.isfinite(snr_db):
        return vocal.astype(np.float32, copy=True)
    rng = rng or np.random.default_rng(0)

    vocal = np.asarray(vocal, dtype=np.float32)
    accomp = np.asarray(accompaniment, dtype=np.float32)
    n = len(vocal)

    if len(accomp) >= n:
        start = int(rng.integers(0, len(accomp) - n + 1))
        accomp = accomp[start:start + n]
    else:
        # Loop with random initial phase so the same accompaniment used twice
        # doesn't always start at sample 0.
        phase = int(rng.integers(0, len(accomp)))
        accomp = np.tile(np.roll(accomp, -phase), int(np.ceil(n / len(accomp))))[:n]

    rms_v = _rms(vocal)
    rms_a = _rms(accomp)
    target_rms_a = rms_v / (10 ** (snr_db / 20.0))
    accomp = accomp * (target_rms_a / max(rms_a, 1e-9))

    mix = vocal + accomp
    peak = np.max(np.abs(mix))
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    return mix.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    accs = prepare_musdb_accompaniments()
    print(f"Prepared {len(accs)} accompaniments:")
    for a in accs:
        x = a.load()
        print(f"  {a.name:40s}  duration={len(x)/SAMPLE_RATE:5.1f}s  "
              f"rms={_rms(x):.3f}  -> {a.audio_path.name}")
