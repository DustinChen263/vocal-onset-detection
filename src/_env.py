"""Environment shims that *must* run before importing numpy/torch/madmom.

Importing this module sets two environment variables that are required for
the vocal-onset env on Apple Silicon:

* ``KMP_DUPLICATE_LIB_OK=TRUE``
    torch ships its own ``libomp.dylib`` and madmom links a second one via
    its build of the system libomp. Without this flag, importing both in the
    same process aborts with::

        OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
        already initialized.

* ``TORCH_HOME=<repo>/.cache/torch``
    Redirects ``torch.hub.load_state_dict_from_url`` away from the default
    ``~/.cache/torch`` (which Cursor's sandbox cannot write to) into a
    workspace-local cache directory.

Always do ``import src._env`` *before* you import torch / demucs / madmom.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_TORCH_CACHE = _PROJECT_ROOT / ".cache" / "torch"
_TORCH_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(_TORCH_CACHE))

# Numba's default cache lives next to the source file inside site-packages,
# which Cursor's sandbox cannot write to. Redirect it inside the workspace.
_NUMBA_CACHE = _PROJECT_ROOT / ".cache" / "numba"
_NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE))
