"""Make ``src/`` importable when the package has not been pip-installed.

Every script imports this first. With ``pip install -e .`` it is a no-op; without
it, the repository still runs straight from a checkout, which keeps the "clone
and run" path working for classroom use.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

__all__ = ["PROJECT_ROOT", "SRC_ROOT"]
