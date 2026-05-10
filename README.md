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
├── README.md                       ← you are here
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/                            ← reusable modules
│   ├── _env.py                     ← env shims (KMP / TORCH_HOME / NUMBA_CACHE_DIR)
│   ├── data.py                     ← Track dataclass + dataset loaders
│   ├── detectors.py                ← three onset detectors
│   ├── evaluate.py                 ← mir_eval wrappers
│   ├── viz.py                      ← spectrogram overlay + error taxonomy
│   ├── mixtures.py                 ← vocadito + MUSDB18 mixture generator      [§7]
│   ├── separation.py               ← demucs vocals-stem wrapper                 [§7]
│   ├── run_separation.py           ← end-to-end source-separation experiment    [§7]
│   └── viz_separation.py           ← plots for the separation extension         [§7]
├── notebooks/
│   └── demo.ipynb                  ← self-contained demonstration
├── results/
│   ├── raw_metrics.csv             ← per-track F / P / R for every (track, detector)
│   ├── error_taxonomy.csv          ← per-track FP / FN breakdown
│   ├── separation_metrics.csv      ← (track, condition, snr_db, detector) → P/R/F  [§7]
│   ├── f_measure_bar.png
│   ├── error_composition.png
│   ├── separation_vocadito_snr.png ← F vs. SNR, mix vs. separated                 [§7]
│   └── separation_dagstuhl.png     ← demucs on Dagstuhl mix vs. original          [§7]
├── tests/
│   └── smoke_test.py               ← quick "does everything import?" check
└── data/                           ← gitignored — populated by mirdata
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
| `Fatal Python error: Segmentation fault ... in _mac_os_check` after installing torch | The pip wheel's bundled OpenBLAS is broken on Apple Silicon. Reinstall numpy from conda-forge: `conda install -c conda-forge numpy=1.23.5 --force-reinstall` |
| `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized` | torch and madmom each ship a libomp. Set `KMP_DUPLICATE_LIB_OK=TRUE` (handled automatically by `src/_env.py`). |
| `RuntimeError: cannot cache function ... no locator available` during `import librosa` | Numba is trying to write to read-only `site-packages`. Set `NUMBA_CACHE_DIR=$repo/.cache/numba` (handled by `src/_env.py`). |
| `RuntimeError: ffmpeg or ffprobe could not be found` from `musdb` / `stempeg` | `conda install -c conda-forge ffmpeg` |
| `ModuleNotFoundError: No module named 'src'` in the notebook | Make sure you launched Jupyter Lab from the project root and that the first notebook cell runs `sys.path.insert(0, os.path.abspath('..'))` |
| `Permission denied to <other-account>` when pushing to GitHub | Clear cached credentials: `printf "protocol=https\nhost=github.com\n\n" \| git credential-osxkeychain erase`, then re-push |

## 7. Optional extension — source separation

> **Question.** *If we run a state-of-the-art source separator on a
> vocal+accompaniment mixture, do downstream onset detectors recover
> their solo-vocal performance?*

We answer this with a controlled experiment on vocadito and a more
realistic test on Dagstuhl ChoirSet:

1. **vocadito + MUSDB18** — for each of the 40 vocadito tracks, mix the
   clean vocal with one randomly-chosen accompaniment from the public
   MUSDB18 sample at three vocal-to-accompaniment SNRs (−5, 0, +5 dB).
   Then run [Hybrid Transformer Demucs](https://github.com/facebookresearch/demucs)
   (`htdemucs`) on the mix and pull out the vocals stem. Evaluate every
   detector on three audio variants per track: `clean`, `mix`, `separated`.
2. **Dagstuhl ChoirSet** — run htdemucs on the original polyphonic
   room-mic mix (no isolated stem available) and compare F-measure on
   `mix_orig` vs. `separated`.

### Headline finding (from `results/separation_metrics.csv`, 960 rows)

Mean F-measure (±50 ms tolerance):

| Dataset | Condition | Spectral Flux | SuperFlux | madmom CNN |
|---|---|---|---|---|
| **vocadito** | clean (40 tracks) | 0.568 | 0.552 | 0.662 |
| | mix @ −5 dB | 0.380 | 0.375 | 0.422 |
| | mix @  0 dB | 0.407 | 0.405 | 0.463 |
| | mix @ +5 dB | 0.432 | 0.435 | 0.497 |
| | **separated @ −5 dB** | **0.530** | **0.543** | **0.660** |
| | **separated @  0 dB** | **0.559** | **0.571** | **0.682** |
| | **separated @ +5 dB** | **0.587** | **0.578** | **0.686** |
| **Dagstuhl ChoirSet** | original mix (20 tracks) | 0.287 | **0.334** | 0.306 |
| | demucs-separated | 0.288 | 0.332 | **0.338** |

Three clean takeaways:

* **On pop-style mixtures, separation almost fully restores — and at
  ≥ 0 dB even slightly *exceeds* — clean-vocal performance.** All three
  detectors at all three SNRs come within ~0.025 F of the clean baseline
  after demucs. madmom CNN at SNR = +5 dB actually goes from clean 0.662
  to separated 0.686 (+0.024). Plausibly because demucs strips away
  pre-attack breath/room noise that the detectors were spuriously firing on.
* **The accompaniment costs you ~0.15 F if you don't separate.** Going
  from clean → mix at 0 dB drops every detector by 0.15–0.20 F-measure;
  demucs recovers all of it.
* **On choir, demucs is a wash.** htdemucs was trained on MUSDB18 (pop
  with a single lead vocalist) so SATB harmonies are out-of-distribution.
  It nudges madmom CNN up by +0.03 F (probably by removing room reverb)
  but does not affect the librosa detectors. The detector ranking stays
  the same: SuperFlux still wins (0.33), and separation does *not* close
  the solo↔polyphonic gap.

### How to reproduce §7

```bash
# Full run (~30 min on M-series CPU)
python -m src.run_separation

# Quick check (2+2 tracks, 1 SNR — 2-3 minutes)
python -m src.run_separation --max-vocadito 2 --max-dagstuhl 2 --snrs 0 \
    --out results/separation_metrics_smoke.csv

# Re-render the two figures from the CSV
python -m src.viz_separation
```

The first run downloads:
* MUSDB18 sample (~150 MB, into `data/musdb_sample/`),
* htdemucs weights (~80 MB, into `.cache/torch/`).

Both are gitignored.

### Apple-Silicon-specific gotchas the env shim handles for you

`src/_env.py` (auto-imported by every separation module) sets:

| env var | why we set it |
|---|---|
| `KMP_DUPLICATE_LIB_OK=TRUE` | torch and madmom each link their own `libomp.dylib`; without this the second import aborts with `OMP: Error #15`. |
| `TORCH_HOME=$repo/.cache/torch` | so `torch.hub` writes inside the workspace (avoids `~/.cache/torch` permission errors). |
| `NUMBA_CACHE_DIR=$repo/.cache/numba` | librosa's numba JIT otherwise tries to cache inside read-only `site-packages`. |

If you see `Fatal Python error: Segmentation fault ... in _mac_os_check`
when importing numpy, the pip wheel's bundled OpenBLAS is broken on your
machine — fix with `conda install -c conda-forge numpy=1.23.5
--force-reinstall`.

## 8. License & citation

This code is released under the MIT License (see `LICENSE`).
The two datasets retain their original licenses:

* vocadito — CC BY 4.0. Bittner et al., "vocadito: A small dataset of solo vocals with f0, note, and lyric annotations." (2021).
* Dagstuhl ChoirSet — CC BY 4.0. Rosenzweig et al., "Dagstuhl ChoirSet: A multitrack dataset for MIR research on choral music." TISMIR 3.1 (2020).

If you use this code, please also cite:

* Böck & Widmer, "Maximum Filter Vibrato Suppression for Onset Detection," DAFx 2013.
* Schlüter & Böck, "Improved Musical Onset Detection with Convolutional Neural Networks," ICASSP 2014.
