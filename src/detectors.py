"""Three onset detectors, all returning ``np.ndarray`` of times in seconds.

To be filled in during Step 4. Each function takes raw mono audio +
sample rate and returns a 1-D array of estimated onset times.
"""

from __future__ import annotations
import numpy as np


def detect_spectral_flux(audio: np.ndarray, sr: int) -> np.ndarray:
    """librosa Spectral Flux baseline. TODO: Step 4."""
    raise NotImplementedError


def detect_superflux(audio: np.ndarray, sr: int) -> np.ndarray:
    """librosa SuperFlux (vibrato-robust). TODO: Step 4."""
    raise NotImplementedError


def detect_madmom_cnn(audio: np.ndarray, sr: int) -> np.ndarray:
    """madmom CNNOnsetProcessor (pre-trained CNN). TODO: Step 4."""
    raise NotImplementedError


DETECTORS = {
    "spectral_flux": detect_spectral_flux,
    "superflux":     detect_superflux,
    "madmom_cnn":    detect_madmom_cnn,
}
