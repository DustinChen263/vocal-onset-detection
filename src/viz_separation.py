"""Plotting helpers for the source-separation extension experiment.

Two figures, both produced from ``results/separation_metrics.csv``:

1. :func:`plot_vocadito_snr_curve` — F-measure vs. SNR for the three
   detectors, with one panel per condition (mix / separated). Adds the
   ``clean`` reference as a horizontal line.

2. :func:`plot_dagstuhl_separation_bars` — paired bars (mix_orig vs.
   separated) for each detector on Dagstuhl ChoirSet.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DETECTOR_ORDER = ["spectral_flux", "superflux", "madmom_cnn"]
DETECTOR_COLORS = {
    "spectral_flux": "#4C72B0",
    "superflux":     "#55A868",
    "madmom_cnn":    "#C44E52",
}


# ---------------------------------------------------------------------------
# vocadito: F-measure vs. SNR (mix and separated), with clean reference
# ---------------------------------------------------------------------------

def plot_vocadito_snr_curve(df: pd.DataFrame,
                            ax: plt.Axes | None = None,
                            figsize=(11, 4.5)) -> plt.Figure:
    """Plot F-measure as a function of vocal-to-accompaniment SNR for
    each detector. Solid lines = ``mix``, dashed lines = ``separated``.
    A horizontal dotted line marks the per-detector ``clean`` reference.
    """
    voc = df[df["dataset"] == "vocadito"].copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    snrs = sorted(voc.loc[voc["snr_db"].notna(), "snr_db"].unique())

    for det in DETECTOR_ORDER:
        sub = voc[voc["detector"] == det]
        clean_f = sub[sub["condition"] == "clean"]["f_measure"].mean()
        ax.axhline(clean_f, color=DETECTOR_COLORS[det],
                   ls=":", lw=1, alpha=0.6)

        mix_means = [sub[(sub["condition"] == "mix") & (sub["snr_db"] == s)]
                     ["f_measure"].mean() for s in snrs]
        sep_means = [sub[(sub["condition"] == "separated") & (sub["snr_db"] == s)]
                     ["f_measure"].mean() for s in snrs]
        mix_stds = [sub[(sub["condition"] == "mix") & (sub["snr_db"] == s)]
                    ["f_measure"].std() for s in snrs]
        sep_stds = [sub[(sub["condition"] == "separated") & (sub["snr_db"] == s)]
                    ["f_measure"].std() for s in snrs]

        ax.errorbar(snrs, mix_means, yerr=mix_stds,
                    color=DETECTOR_COLORS[det], marker="o", ls="-",
                    capsize=3, label=f"{det} (mix)")
        ax.errorbar(snrs, sep_means, yerr=sep_stds,
                    color=DETECTOR_COLORS[det], marker="s", ls="--",
                    capsize=3, label=f"{det} (separated)")

    ax.set_xlabel("Vocal-to-accompaniment SNR (dB)")
    ax.set_ylabel("F-measure (±50 ms)")
    ax.set_title("vocadito: source separation almost fully restores "
                 "clean-vocal performance")
    ax.set_xticks(snrs)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2,
              framealpha=0.9, title="dotted = clean reference")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dagstuhl: paired bars (mix_orig vs. separated)
# ---------------------------------------------------------------------------

def plot_dagstuhl_separation_bars(df: pd.DataFrame,
                                  ax: plt.Axes | None = None,
                                  figsize=(8, 4.5)) -> plt.Figure:
    """Bar chart of F-measure for ``mix_orig`` vs. ``separated`` on
    Dagstuhl ChoirSet, grouped by detector."""
    dcs = df[(df["dataset"] == "dagstuhl_choirset")
             & df["condition"].isin(["mix_orig", "separated"])].copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    detectors = DETECTOR_ORDER
    conditions = ["mix_orig", "separated"]
    x = np.arange(len(detectors))
    w = 0.35

    for i, cond in enumerate(conditions):
        means = [dcs[(dcs["detector"] == d) & (dcs["condition"] == cond)]
                 ["f_measure"].mean() for d in detectors]
        stds = [dcs[(dcs["detector"] == d) & (dcs["condition"] == cond)]
                ["f_measure"].std() for d in detectors]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4,
               label="original room-mic mix" if cond == "mix_orig"
                     else "demucs-separated vocals",
               color="#888" if cond == "mix_orig" else "#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(detectors)
    ax.set_ylabel("F-measure (±50 ms)")
    ax.set_title("Dagstuhl ChoirSet: demucs (trained on pop) is roughly "
                 "neutral on choir mixtures")
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: end-to-end CSV → 2 PNGs
# ---------------------------------------------------------------------------

def make_separation_plots(csv_path: str | Path,
                          out_dir: str | Path,
                          dpi: int = 150) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    fig1 = plot_vocadito_snr_curve(df)
    p1 = out_dir / "separation_vocadito_snr.png"
    fig1.savefig(p1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plot_dagstuhl_separation_bars(df)
    p2 = out_dir / "separation_dagstuhl.png"
    fig2.savefig(p2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)

    return {"vocadito": p1, "dagstuhl": p2}


if __name__ == "__main__":
    from src.data import PROJECT_ROOT
    paths = make_separation_plots(
        PROJECT_ROOT / "results" / "separation_metrics.csv",
        PROJECT_ROOT / "results")
    for k, v in paths.items():
        print(f"{k}: {v}")
