"""Torch device selection.

Training here is dominated by tiny MLP forward passes and Python-side
environment stepping, so CPU is frequently *faster* than GPU for the default
network size. ``resolve_device("auto")`` therefore prefers CUDA only when it is
actually available, and every script exposes ``--device`` to force a choice.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

import torch

__all__ = ["resolve_device", "DeviceInfo", "describe_device"]


@dataclass(frozen=True)
class DeviceInfo:
    """Human-readable description of the compute device, recorded in run metadata."""

    device: str
    name: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    platform: str

    def as_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "name": self.name,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "platform": self.platform,
        }


def resolve_device(preference: str = "auto") -> torch.device:
    """Return a usable ``torch.device`` for ``preference``.

    ``"auto"`` selects CUDA when available and CPU otherwise. An explicit
    ``"cuda"`` request on a machine without CUDA raises, because silently
    training on CPU when the user asked for GPU hides a real problem.
    """
    choice = (preference or "auto").strip().lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch or use --device cpu."
            )
        return torch.device(choice)
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference: {preference!r}")


def describe_device(device: torch.device) -> DeviceInfo:
    """Collect device/runtime facts for the reproducibility record."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    else:
        name = platform.processor() or "cpu"
    return DeviceInfo(
        device=str(device),
        name=name,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        platform=platform.platform(),
    )
