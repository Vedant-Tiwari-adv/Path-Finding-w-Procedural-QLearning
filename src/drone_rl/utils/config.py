"""YAML configuration loading with include-merging and dotted-key overrides.

Experiments are configuration-driven: every tunable (reward weight, city size,
DQN hyper-parameter) lives in ``configs/*.yaml`` rather than in Python source.
A config may pull in other files via a top-level ``defaults`` list so that, for
example, ``dqn.yaml`` can reuse ``environment.yaml`` without duplicating it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = ["load_config", "deep_merge", "apply_overrides", "save_config", "get_in"]


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into ``base``, returning a new dict.

    Nested mappings are merged key-by-key; every other type (including lists) is
    replaced wholesale by the value from ``update``.
    """
    merged: dict[str, Any] = dict(copy.deepcopy(dict(base)))
    for key, value in update.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce_scalar(text: str) -> Any:
    """Parse a command-line override value using YAML scalar rules."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` strings on top of a config, returning a new dict.

    Used by every script's ``--set`` flag so a single YAML file can be reused for
    ablations without editing it (``--set dqn.learning_rate=1e-4``).
    """
    result = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must look like 'section.key=value', got {override!r}")
        dotted_key, _, raw_value = override.partition("=")
        keys = dotted_key.strip().split(".")
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            node = cursor.get(key)
            if not isinstance(node, dict):
                node = {}
                cursor[key] = node
            cursor = node
        cursor[keys[-1]] = _coerce_scalar(raw_value.strip())
    return result


def get_in(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read ``config['a']['b']`` via the dotted path ``"a.b"``."""
    cursor: Any = config
    for key in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Load a YAML config, resolving its ``defaults:`` includes and overrides.

    ``defaults`` entries are paths relative to the including file. They are
    merged in listed order, and the including file always wins over its
    defaults. Cyclic includes raise ``ValueError``.
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        raise ValueError(f"Cyclic config include detected at {config_path}")
    seen.add(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a mapping, got {type(raw).__name__} in {config_path}")

    includes = raw.pop("defaults", []) or []
    if isinstance(includes, str):
        includes = [includes]

    merged: dict[str, Any] = {}
    for include in includes:
        include_path = (config_path.parent / include).resolve()
        merged = deep_merge(merged, load_config(include_path, _seen=seen))
    merged = deep_merge(merged, raw)

    if overrides:
        merged = apply_overrides(merged, overrides)
    return merged


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write a config to disk as YAML, creating parent directories."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, default_flow_style=False)
    return out_path
