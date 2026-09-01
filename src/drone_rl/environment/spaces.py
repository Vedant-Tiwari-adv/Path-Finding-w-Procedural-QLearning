"""Phase 4 - the discrete action space.

Actions are declared as data (:class:`ActionSpec`) rather than hard-coded
``if`` branches, so the action set can be extended or restricted from YAML
without touching the dynamics code. Each action maps to a body-frame velocity
command plus a yaw-rate command; the drone integrates those in
``drone.apply_action``.

Body frame convention
---------------------
``+x`` is forward along the drone's heading, ``+y`` is to its left, ``+z`` is
up. Yaw is measured counter-clockwise from the world ``+x`` axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["ActionSpec", "ActionSet", "DEFAULT_ACTIONS", "build_action_set"]


@dataclass(frozen=True)
class ActionSpec:
    """One discrete action.

    Attributes
    ----------
    name:
        Human-readable label used in logs, the GUI overlay and debug output.
    forward, left, up:
        Body-frame velocity command components, each in ``[-1, 1]`` and scaled
        by the drone's ``max_speed``.
    yaw_rate:
        Yaw command in ``[-1, 1]``, scaled by the drone's ``max_yaw_rate``.
    energy_scale:
        Relative energy draw. Hovering still costs energy (the rotors must
        counteract gravity), so this multiplies a manoeuvre cost that is added
        on top of a constant hover draw.
    """

    name: str
    forward: float = 0.0
    left: float = 0.0
    up: float = 0.0
    yaw_rate: float = 0.0
    energy_scale: float = 1.0

    @property
    def body_velocity(self) -> np.ndarray:
        """Body-frame velocity command as ``[forward, left, up]``."""
        return np.array([self.forward, self.left, self.up], dtype=np.float64)


#: Default nine-action set from the project specification.
DEFAULT_ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec("hover", energy_scale=0.35),
    ActionSpec("forward", forward=1.0, energy_scale=1.0),
    ActionSpec("backward", forward=-1.0, energy_scale=1.0),
    ActionSpec("left", left=1.0, energy_scale=1.0),
    ActionSpec("right", left=-1.0, energy_scale=1.0),
    ActionSpec("ascend", up=1.0, energy_scale=1.6),
    ActionSpec("descend", up=-1.0, energy_scale=0.7),
    ActionSpec("yaw_left", yaw_rate=1.0, energy_scale=0.6),
    ActionSpec("yaw_right", yaw_rate=-1.0, energy_scale=0.6),
)


class ActionSet:
    """An ordered collection of :class:`ActionSpec` with index lookup."""

    def __init__(self, actions: Sequence[ActionSpec]) -> None:
        if not actions:
            raise ValueError("Action set must contain at least one action")
        self._actions = tuple(actions)
        self._by_name = {a.name: i for i, a in enumerate(self._actions)}
        if len(self._by_name) != len(self._actions):
            raise ValueError("Action names must be unique")

    def __len__(self) -> int:
        return len(self._actions)

    def __getitem__(self, index: int) -> ActionSpec:
        return self._actions[index]

    def __iter__(self):
        return iter(self._actions)

    @property
    def names(self) -> tuple[str, ...]:
        """Action names in index order."""
        return tuple(a.name for a in self._actions)

    def index(self, name: str) -> int:
        """Return the discrete index of the action called ``name``."""
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown action {name!r}; available: {self.names}") from exc


def build_action_set(config: Mapping[str, Any] | None = None) -> ActionSet:
    """Build an :class:`ActionSet` from the ``actions:`` YAML block.

    With no config (or an empty one) the default nine-action set is used. A
    config may either restrict the defaults by name::

        actions:
          use: [hover, forward, left, right, ascend, descend]

    or define actions explicitly::

        actions:
          custom:
            - {name: forward, forward: 1.0, energy_scale: 1.0}
    """
    if not config:
        return ActionSet(DEFAULT_ACTIONS)

    custom = config.get("custom")
    if custom:
        return ActionSet([ActionSpec(**dict(entry)) for entry in custom])

    selected = config.get("use")
    if selected:
        by_name = {a.name: a for a in DEFAULT_ACTIONS}
        missing = [n for n in selected if n not in by_name]
        if missing:
            raise KeyError(f"Unknown action names in config: {missing}")
        return ActionSet([by_name[n] for n in selected])

    return ActionSet(DEFAULT_ACTIONS)
