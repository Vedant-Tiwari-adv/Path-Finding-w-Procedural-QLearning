"""Seeding and experiment provenance.

Two separable concerns live here:

* **Process seeding** (``set_global_seeds``) makes a *training run* repeatable.
* **City / episode seeding** is handled by the environment itself via explicit
  ``numpy.random.Generator`` objects, so an episode is reproducible regardless
  of global state. Nothing in the environment reads the global NumPy seed.

``ExperimentRecord`` captures everything the project requires to be stored
alongside a checkpoint: seeds, configs, library versions, device, timestamp.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

__all__ = ["set_global_seeds", "ExperimentRecord", "seed_ranges_from_config", "SeedRanges"]


def set_global_seeds(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy and Torch RNGs for the current process.

    ``deterministic_torch`` additionally forces cuDNN into deterministic mode,
    which costs throughput and is therefore off by default.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class SeedRanges:
    """Disjoint city-seed pools for train / validation / test.

    Keeping these disjoint is what makes the generalisation claim meaningful:
    the agent is evaluated on cities whose layouts it has provably never seen.
    """

    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]

    def __post_init__(self) -> None:
        spans = {"train": self.train, "validation": self.validation, "test": self.test}
        for name, (lo, hi) in spans.items():
            if hi < lo:
                raise ValueError(f"{name} seed range is empty: [{lo}, {hi}]")
        ordered = sorted(spans.items(), key=lambda kv: kv[1][0])
        for (name_a, (_, hi_a)), (name_b, (lo_b, _)) in zip(ordered, ordered[1:]):
            if lo_b <= hi_a:
                raise ValueError(
                    f"Seed ranges {name_a} and {name_b} overlap; test seeds must never "
                    "leak into training."
                )

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


def seed_ranges_from_config(city_config: Mapping[str, Any]) -> SeedRanges:
    """Build :class:`SeedRanges` from the ``city:`` block of a config."""
    return SeedRanges(
        train=(int(city_config["train_seed_start"]), int(city_config["train_seed_end"])),
        validation=(
            int(city_config["validation_seed_start"]),
            int(city_config["validation_seed_end"]),
        ),
        test=(int(city_config["test_seed_start"]), int(city_config["test_seed_end"])),
    )


def _git_revision() -> str | None:
    """Return the current git commit, or ``None`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@dataclass
class ExperimentRecord:
    """Full provenance for one experiment, saved next to its checkpoints."""

    name: str
    seed: int
    config: dict[str, Any]
    device: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    torch_version: str = field(default_factory=lambda: torch.__version__)
    numpy_version: str = field(default_factory=lambda: np.__version__)
    platform: str = field(default_factory=platform.platform)
    git_commit: str | None = field(default_factory=_git_revision)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Write the record as pretty-printed JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2, default=str)
        return out_path
