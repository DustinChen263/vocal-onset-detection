"""Dataset loaders for vocadito and Dagstuhl ChoirSet.

Both datasets are wrapped behind a uniform ``Track`` interface so that the
rest of the pipeline does not care which source the data came from.

Usage
-----
>>> from src.data import load_vocadito, load_dagstuhl
>>> tracks = load_vocadito(data_home="data/vocadito")
>>> tr = tracks[0]
>>> tr.audio              # numpy array, mono, 22050 Hz
>>> tr.onsets             # numpy array of ground-truth onset times (sec)
>>> tr.dataset, tr.track_id
('vocadito', 'vocadito_1')
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Track:
    """Uniform container for an audio clip + its onset ground truth."""
    dataset: str
    track_id: str
    audio: np.ndarray        # mono, float32
    sample_rate: int
    onsets: np.ndarray       # ground-truth onset times in seconds
    metadata: dict           # anything dataset-specific (language, voice part, mic type, ...)


def load_vocadito(data_home: str = "data/vocadito") -> list[Track]:
    """Load all vocadito tracks. TODO: implement in Step 3."""
    raise NotImplementedError


def load_dagstuhl(data_home: str = "data/dagstuhl_choirset",
                  mic_type: str = "HSM") -> list[Track]:
    """Load Dagstuhl ChoirSet tracks for a given mic type. TODO: Step 3."""
    raise NotImplementedError
