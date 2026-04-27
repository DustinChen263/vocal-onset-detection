# Vocal Onset Detection: How well do classical onset detectors work on singing voice?

> Final project for **Music Information Retrieval**.
> All code, data loaders, evaluation, and the demo notebook are in this
> repository. The full analysis runs end-to-end with one command (see
> [§ Reproducing the results](#reproducing-the-results)).

## 1. Research question

> Across **solo vocals** (vocadito) and **polyphonic choir vocals**
> (Dagstuhl ChoirSet), how do three classical onset detectors —
> librosa Spectral Flux, librosa SuperFlux, and madmom's CNN-based
> detector — compare in terms of F-measure, and what kinds of errors
> dominate (missed legato transitions vs. vibrato false positives)?

## 2. Headline results

F-measure with the MIREX-standard ±50 ms tolerance (`mir_eval.onset.f_measure`),
mean over all tracks ± 1 standard deviation:

| Dataset | Spectral Flux | SuperFlux | madmom CNN |
|---|---|---|---|
| **vocadito** (solo, 40 tracks) | 0.568 ± 0.067 | 0.552 ± 0.094 | **0.662 ± 0.087** |
| **Dagstuhl ChoirSet** (polyphonic, 20 tracks) | 0.328 ± 0.057 | **0.337 ± 0.066** | 0.306 ± 0.083 |

Three findings:

1. **Polyphonic choir is much harder.** Best F drops from 0.66 (solo) to
   0.34 (choir) — performance halves.
2. **The winner flips between datasets.** madmom's CNN is best on solo
   vocals and **worst** on the polyphonic mix.
3. **CNN's failure mode on choir is low recall** (0.235): it rarely fires
   spuriously but misses ~76 % of true onsets. SuperFlux's vibrato-robust
   spectral flux is more balanced and wins on F-measure.

See `notebooks/demo.ipynb` for the full analysis with spectrogram overlays
and an error taxonomy.

## 3. Datasets

| Dataset | Type | # Tracks | Annotation used |
|---|---|---|---|
| [vocadito](https://zenodo.org/record/5578807) | Solo vocals (multi-language, multi-style) | 40 | manual notes — annotator A1 |
| [Dagstuhl ChoirSet](https://zenodo.org/record/4543957) | SATB choir, room-mic mix | 20 multitracks of *Locus Iste* | per-multitrack score notes (deduplicated within 30 ms) |

Both are downloaded automatically via [`mirdata`](https://mirdata.readthedocs.io/).
Total ≈ 5 GB, written to `data/` (gitignored).

## 4. Methods (baselines)

| Method | Library | Idea |
|---|---|---|
| **Spectral Flux** | `librosa` | Sum of positive frame-to-frame log-magnitude differences. |
| **SuperFlux** | `librosa` | Spectral flux + 3-bin maximum filter on the magnitude spectrogram (Böck & Widmer, DAFx 2013). Designed to suppress vibrato. |
| **madmom CNN** | `madmom` | Pre-trained CNN that outputs a per-frame onset probability at 100 fps (Schlüter & Böck, 2014). |

All three share a uniform interface:

```python
def detect(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return onset times in seconds."""
```

so the evaluation pipeline is detector-agnostic (`src/evaluate.py`).

## 5. Repository structure

```
vocal-onset-detection/
├── README.md                  ← you are here
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/                       ← reusable modules
│   ├── data.py                ← Track dataclass + dataset loaders
│   ├── detectors.py           ← three onset detectors
│   ├── evaluate.py            ← mir_eval wrappers
│   └── viz.py                 ← spectrogram overlay + error taxonomy
├── notebooks/
│   └── demo.ipynb             ← self-contained demonstration
├── results/
│   ├── raw_metrics.csv        ← per-track F / P / R for every (track, detector)
│   ├── error_taxonomy.csv     ← per-track FP / FN breakdown
│   ├── f_measure_bar.png
│   └── error_composition.png
├── tests/
│   └── smoke_test.py          ← quick "does everything import?" check
└── data/                      ← gitignored — populated by mirdata
```

## 6. Reproducing the results

### 6.1 Environment

> **Apple Silicon note.** `madmom` (the CNN baseline) was last released in
> 2018 and pre-dates Apple Silicon. The recipe below works on M-series Macs;
> see [§ 6.4](#64-troubleshooting) if your install fails.

```bash
# 1. Clone
git clone https://github.com/DustinChen263/vocal-onset-detection.git
cd vocal-onset-detection

# 2. Create a Python 3.9 environment (madmom does not work on 3.10+)
conda create -n vocal-onset python=3.9 -y
conda activate vocal-onset

# 3. Install scientific stack via conda for stability on Apple Silicon
conda install -c conda-forge \
    numpy=1.23.5 scipy=1.10.1 pandas=2.0.3 matplotlib=3.7.5 -y

# 4. Install audio + MIR libraries
pip install librosa==0.10.1 soundfile==0.12.1 \
            mir_eval==0.7 mirdata==0.3.9 \
            jupyterlab==4.0.11 ipykernel==6.29.0

# 5. Install madmom (needs --no-build-isolation on Apple Silicon)
pip install Cython==0.29.36
pip install --no-build-isolation madmom==0.16.1
```

### 6.2 Smoke test

```bash
python tests/smoke_test.py
```

Should print "All systems go ✓".

### 6.3 Full pipeline

```bash
# Download both datasets (~5 GB, 15-25 min depending on bandwidth)
python -m src.data --download

# Run the full evaluation (3 detectors × 60 tracks, ~1 minute)
python -m src.evaluate

# Or open the demo notebook for the interactive analysis
jupyter lab notebooks/demo.ipynb
```

### 6.4 Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'Cython'` while building madmom | Add `--no-build-isolation`: `pip install --no-build-isolation madmom==0.16.1` |
| `AttributeError: module 'numpy' has no attribute 'float'` when importing madmom | Downgrade numpy: `pip install numpy==1.23.5` |
| `ModuleNotFoundError: No module named 'src'` in the notebook | Make sure you launched Jupyter Lab from the project root and that the first notebook cell runs `sys.path.insert(0, os.path.abspath('..'))` |
| `Permission denied to <other-account>` when pushing to GitHub | Clear cached credentials: `printf "protocol=https\nhost=github.com\n\n" \| git credential-osxkeychain erase`, then re-push |

## 7. License & citation

This code is released under the MIT License (see `LICENSE`).
The two datasets retain their original licenses:

* vocadito — CC BY 4.0. Bittner et al., "vocadito: A small dataset of solo vocals with f0, note, and lyric annotations." (2021).
* Dagstuhl ChoirSet — CC BY 4.0. Rosenzweig et al., "Dagstuhl ChoirSet: A multitrack dataset for MIR research on choral music." TISMIR 3.1 (2020).

If you use this code, please also cite:

* Böck & Widmer, "Maximum Filter Vibrato Suppression for Onset Detection," DAFx 2013.
* Schlüter & Böck, "Improved Musical Onset Detection with Convolutional Neural Networks," ICASSP 2014.
