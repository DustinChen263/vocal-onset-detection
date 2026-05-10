"""Quick sanity check — verifies the environment is correctly set up.

Run::

    python tests/smoke_test.py

Exits with code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import traceback

# Make the project root importable regardless of where this is run from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check(label, fn):
    try:
        fn()
    except Exception as e:
        print(f"  [FAIL] {label}: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        return False
    print(f"  [ ok ] {label}")
    return True


def main() -> int:
    print("Smoke-testing the vocal-onset-detection environment ...\n")

    ok = True

    # 1. Core libs
    def _imports():
        import numpy, scipy, pandas, matplotlib, librosa, mir_eval, mirdata
        from madmom.features.onsets import CNNOnsetProcessor  # noqa: F401
    ok &= check("imports (numpy, librosa, mir_eval, mirdata, madmom, ...)",
                _imports)

    # 2. Pinned numpy version (madmom needs <1.24)
    def _numpy_version():
        import numpy as np
        major, minor = map(int, np.__version__.split(".")[:2])
        assert (major, minor) <= (1, 23), (
            f"numpy is {np.__version__}; downgrade to 1.23.x for madmom")
    ok &= check("numpy version compatible with madmom", _numpy_version)

    # 3. Project modules import
    def _project():
        from src.data      import Track, load_vocadito, load_dagstuhl  # noqa
        from src.detectors import DETECTORS                             # noqa
        from src.evaluate  import evaluate_track, evaluate_all          # noqa
        from src.viz       import plot_onsets_on_spectrogram            # noqa
    ok &= check("project modules importable", _project)

    # 4. madmom CNN can be loaded
    def _cnn():
        from src.detectors import _get_madmom_processors
        cnn, picker = _get_madmom_processors()
        assert cnn is not None and picker is not None
    ok &= check("madmom CNN weights load", _cnn)

    # 5. Source-separation extension modules (optional but ship in the repo)
    def _sep():
        import src._env  # noqa: F401
        from src.mixtures   import (mix_at_snr,                        # noqa
                                    prepare_musdb_accompaniments,
                                    SAMPLE_RATE)
        from src.separation import separate_vocals                      # noqa
        from src.viz_separation import (plot_vocadito_snr_curve,        # noqa
                                        plot_dagstuhl_separation_bars)
    ok &= check("source-separation extension importable", _sep)

    print()
    if ok:
        print("All systems go ✓")
        return 0
    else:
        print("One or more checks failed — see trace above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
