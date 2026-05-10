"""Source-separation wrapper around demucs (Hybrid Transformer Demucs).

The optional half of the project asks: *if we run a state-of-the-art source
separator on a vocal+accompaniment mixture, do downstream onset detectors
recover their solo-vocal performance?* This module exposes a single
function — :func:`separate_vocals` — and lazily loads the htdemucs model on
first use so we pay the ~80 MB / 5 s setup cost only once per process.
"""

from __future__ import annotations

import src._env  # noqa: F401  — env shims (KMP, TORCH_HOME, NUMBA_CACHE_DIR)

from typing import Optional
import numpy as np
import librosa
import torch


_MODEL = None
_DEVICE: Optional[torch.device] = None


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_model():
    """Lazy-load htdemucs once per process."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        from demucs.pretrained import get_model
        _DEVICE = _pick_device()
        _MODEL = get_model("htdemucs").to(_DEVICE).eval()
    return _MODEL


def separate_vocals(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Run demucs and return the *vocal stem* as mono audio.

    Parameters
    ----------
    audio : 1-D mono ``float32`` or ``float64`` array.
        Will be resampled to 44.1 kHz and duplicated to stereo internally
        because htdemucs is trained at (stereo, 44.1 kHz).
    sr : input sample rate.

    Returns
    -------
    vocals_mono : 1-D ``float32`` array
    sr_out : 44100
    """
    from demucs.apply import apply_model

    model = _get_model()
    target_sr = model.samplerate          # 44100

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # (channels, samples) — duplicate mono to stereo (htdemucs needs 2 ch).
    if audio.ndim == 1:
        audio_2ch = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2 and audio.shape[0] in (1, 2):
        audio_2ch = audio if audio.shape[0] == 2 else np.repeat(audio, 2, axis=0)
    else:
        raise ValueError(f"audio must be mono or (channels, samples); got {audio.shape}")

    x = torch.from_numpy(audio_2ch.astype(np.float32))[None]   # (1, 2, T)
    x = x.to(_DEVICE)

    with torch.no_grad():
        sources = apply_model(model, x, device=str(_DEVICE), progress=False)
    # sources: (1, n_sources, 2, T). model.sources is e.g.
    # ['drums', 'bass', 'other', 'vocals'].
    vocals_idx = model.sources.index("vocals")
    vocals = sources[0, vocals_idx].cpu().numpy()              # (2, T)
    vocals_mono = vocals.mean(axis=0).astype(np.float32)
    return vocals_mono, target_sr


if __name__ == "__main__":
    # Quick smoke: separate the first vocadito track + a random accompaniment.
    import argparse, time
    from src.data import load_vocadito
    from src.mixtures import prepare_musdb_accompaniments, mix_at_snr, SAMPLE_RATE

    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=0.0)
    args = parser.parse_args()

    tr = load_vocadito()[0]
    audio, sr = tr.load_audio(sr=SAMPLE_RATE)
    accs = prepare_musdb_accompaniments()
    acc = accs[0].load(sr=SAMPLE_RATE)

    mix = mix_at_snr(audio, acc, snr_db=args.snr)
    print(f"Track {tr.track_id}: {len(audio)/sr:.1f}s, mixing @ SNR={args.snr} dB ...")

    t0 = time.time()
    voc, _ = separate_vocals(mix, sr)
    print(f"separated in {time.time()-t0:.1f}s; vocals shape={voc.shape}")
