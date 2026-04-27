"""Three onset detectors with a uniform interface.

Each detector takes mono audio + its sample rate and returns a 1-D
``np.ndarray`` of estimated onset times **in seconds**.

Usage
-----
>>> from src.detectors import DETECTORS
>>> onsets = DETECTORS['superflux'](audio, sr)
"""

from __future__ import annotations

import warnings
import numpy as np
import librosa


# ---------------------------------------------------------------------------
# Common hyper-parameters
# ---------------------------------------------------------------------------
# librosa defaults are fine for most music but for vocals these tend to give
# better results. Tweaked with vocadito's 22050 Hz / 44100 Hz sample rates.
HOP_LENGTH = 512        # ~23 ms at 22050 Hz, ~11.6 ms at 44100 Hz
N_FFT      = 2048


# ---------------------------------------------------------------------------
# 1. Spectral Flux  (librosa default)
# ---------------------------------------------------------------------------

def detect_spectral_flux(audio: np.ndarray, sr: int) -> np.ndarray:
    """Classic spectral flux onset detection.

    Onset strength = sum over frequency bins of positive frame-to-frame
    log-magnitude differences. Peak picking by ``librosa.onset.onset_detect``.
    """
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sr,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr,
        hop_length=HOP_LENGTH, units="frames",
    )
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)


# ---------------------------------------------------------------------------
# 2. SuperFlux  (Böck & Widmer, 2013)
# ---------------------------------------------------------------------------

def detect_superflux(audio: np.ndarray, sr: int) -> np.ndarray:
    """SuperFlux: spectral flux + maximum filter to suppress vibrato.

    Reference: Sebastian Böck & Gerhard Widmer, "Maximum Filter Vibrato
    Suppression for Onset Detection" (DAFx 2013).

    Implementation note: librosa exposes SuperFlux through the
    ``onset_strength`` interface by setting ``lag=2`` (compute flux against
    the frame 2 ago, not 1) and ``max_size`` (frequency-axis max filter).
    """
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sr,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        lag=2,
        max_size=3,
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr,
        hop_length=HOP_LENGTH, units="frames",
        # SuperFlux benefits from slightly tighter peak picking
        pre_max=int(0.030 * sr / HOP_LENGTH),    # 30 ms
        post_max=int(0.030 * sr / HOP_LENGTH),
        pre_avg=int(0.100 * sr / HOP_LENGTH),    # 100 ms
        post_avg=int(0.100 * sr / HOP_LENGTH),
        wait=int(0.030 * sr / HOP_LENGTH),       # min separation 30 ms
        delta=0.07,                              # threshold above local mean
    )
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)


# ---------------------------------------------------------------------------
# 3. madmom CNN  (Schlüter & Böck, 2014)
# ---------------------------------------------------------------------------

# Lazy global so we only load the CNN weights once (slow first time).
_madmom_cnn = None
_madmom_picker = None


def _get_madmom_processors():
    global _madmom_cnn, _madmom_picker
    if _madmom_cnn is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from madmom.features.onsets import (
                CNNOnsetProcessor, OnsetPeakPickingProcessor)
        _madmom_cnn = CNNOnsetProcessor()
        # The CNN is trained at 100 fps; threshold tuned on its training set.
        _madmom_picker = OnsetPeakPickingProcessor(
            threshold=0.5, smooth=0.0, fps=100)
    return _madmom_cnn, _madmom_picker


def detect_madmom_cnn(audio: np.ndarray, sr: int) -> np.ndarray:
    """Pre-trained CNN onset detector (madmom).

    madmom internally expects audio at 44100 Hz, so we resample if needed.
    """
    cnn, picker = _get_madmom_processors()

    if sr != 44100:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        sr = 44100

    audio = audio.astype(np.float32, copy=False)
    activation = cnn(audio)            # 1-D array of onset probabilities @ 100 fps
    onsets = picker(activation)        # array of times (sec)
    return np.asarray(onsets, dtype=float)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DETECTORS = {
    "spectral_flux": detect_spectral_flux,
    "superflux":     detect_superflux,
    "madmom_cnn":    detect_madmom_cnn,
}


if __name__ == "__main__":
    # Smoke test on one vocadito track
    from src.data import load_vocadito

    tr = load_vocadito()[0]
    audio, sr = tr.load_audio()

    print(f"Track: {tr}")
    print(f"  audio: shape={audio.shape}, sr={sr}")
    print(f"  ground-truth onsets: {len(tr.onsets)}")
    print()

    for name, fn in DETECTORS.items():
        est = fn(audio, sr)
        print(f"{name:15s} -> {len(est):3d} onsets, "
              f"first 5: {np.round(est[:5], 3)}")
