# Vocal Onset Detection: How well do classical onset detectors work on singing voice?

> Final project for **Music Information Retrieval**
> Author: *(your name here)*

## 1. Research Question

Onset detection algorithms are typically designed and benchmarked on percussive
or polyphonic instrumental music. **How well do classical onset detection
methods perform on singing voice?** Specifically, we ask:

> Across **solo vocals (vocadito)** and **polyphonic choir vocals
> (Dagstuhl ChoirSet)**, how do three classical onset detectors —
> librosa Spectral Flux, librosa SuperFlux, and madmom's CNN-based
> detector — compare in terms of F-measure, and what kinds of errors
> dominate (missed soft onsets vs. false positives from vibrato)?

## 2. Datasets

| Dataset | Type | # Tracks | Annotations used |
|---|---|---|---|
| [vocadito](https://zenodo.org/record/5578807) | Solo vocals, multi-language | 40 | `notes_manual` onsets |
| [Dagstuhl ChoirSet](https://zenodo.org/record/4543957) | Polyphonic SATB choir | ~55 takes | Per-singer note onsets |

Both datasets are loaded via [`mirdata`](https://mirdata.readthedocs.io/).

## 3. Methods (baselines)

| Method | Library | Idea |
|---|---|---|
| **Spectral Flux** | `librosa` | Frame-to-frame spectral magnitude difference |
| **SuperFlux** | `librosa` | Spectral flux with frequency-trajectory smoothing (robust to vibrato) |
| **CNN** | `madmom` | Pre-trained convolutional onset detector |

## 4. Evaluation

- Metric: `mir_eval.onset.f_measure` with a **±50 ms** tolerance window (MIREX standard)
- Reported per-track and aggregated per dataset
- Qualitative error analysis: spectrogram overlays + error categorization
  (missed legato transitions vs. vibrato false positives vs. consonant double-trigger)

## 5. Repository structure

```
vocal-onset-detection/
├── README.md              ← you are here
├── requirements.txt       ← Python dependencies
├── .gitignore             ← files git should ignore
├── src/                   ← reusable Python modules
│   ├── data.py            ← dataset loaders
│   ├── detectors.py       ← the 3 onset detectors
│   ├── evaluate.py        ← mir_eval wrappers
│   └── viz.py             ← plotting helpers
├── notebooks/
│   └── demo.ipynb         ← self-contained demonstration
├── data/                  ← (gitignored) downloaded audio + annotations
├── results/               ← evaluation tables & figures
└── tests/                 ← small sanity-check scripts
```

## 6. How to reproduce

```bash
# 1. Clone
git clone https://github.com/<your-username>/vocal-onset-detection.git
cd vocal-onset-detection

# 2. Create a clean Python 3.9 environment
conda create -n vocal-onset python=3.9 -y
conda activate vocal-onset

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets (handled by mirdata, ~5 GB total)
python -m src.data --download

# 5. Run the notebook
jupyter lab notebooks/demo.ipynb
```

## 7. Results

*(filled in once experiments are complete)*

## 8. License & citation

Code: MIT.
Datasets retain their original licenses (CC BY 4.0 for both vocadito and
Dagstuhl ChoirSet).
