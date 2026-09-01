"""Q-network architectures (Phase 8).

The baseline is a plain MLP: ``obs -> Linear -> ReLU -> Linear -> ReLU ->
Q-values``, one output per discrete action, exactly as specified. A dueling
head is available behind a config flag but is **off** for the baseline so the
comparison stays honest.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

__all__ = ["QNetwork", "build_activation"]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


def build_activation(name: str) -> nn.Module:
    """Instantiate an activation module by name."""
    key = name.strip().lower()
    if key not in _ACTIVATIONS:
        raise KeyError(f"Unknown activation {name!r}; available: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


class QNetwork(nn.Module):
    """State-action value network ``Q(s, .)`` over a discrete action set.

    Parameters
    ----------
    observation_size:
        Length of the flat observation vector.
    num_actions:
        Size of the discrete action space; equals the number of outputs.
    hidden_sizes:
        Widths of the hidden layers.
    activation:
        Name of the activation used between hidden layers.
    dueling:
        When ``True``, split into value and advantage streams and recombine as
        ``Q = V + (A - mean(A))``. Defaults to ``False`` (baseline MLP).
    layer_norm:
        When ``True``, insert :class:`~torch.nn.LayerNorm` after each hidden
        linear layer. Defaults to ``False`` so the baseline architecture is
        untouched; the proposed agent enables it, because normalising the
        features bounds the scale of the regression target the Bellman update
        chases and is a cheap guard against the value drift the baseline run
        exhibited.
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
        dueling: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if observation_size <= 0 or num_actions <= 0:
            raise ValueError("observation_size and num_actions must be positive")

        self.observation_size = int(observation_size)
        self.num_actions = int(num_actions)
        self.dueling = bool(dueling)
        self.layer_norm = bool(layer_norm)

        layers: list[nn.Module] = []
        in_features = self.observation_size
        for width in hidden_sizes:
            layers.append(nn.Linear(in_features, int(width)))
            if self.layer_norm:
                layers.append(nn.LayerNorm(int(width)))
            layers.append(build_activation(activation))
            in_features = int(width)
        self.feature_extractor = nn.Sequential(*layers)

        if self.dueling:
            self.value_head = nn.Linear(in_features, 1)
            self.advantage_head = nn.Linear(in_features, self.num_actions)
        else:
            self.output_head = nn.Linear(in_features, self.num_actions)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Orthogonal init with a small gain; standard and stable for DQN MLPs."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape ``(batch, num_actions)``."""
        features = self.feature_extractor(observations)
        if self.dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            # Subtracting the mean advantage fixes the value/advantage gauge.
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.output_head(features)

    @torch.no_grad()
    def greedy_action(self, observation: torch.Tensor) -> int:
        """Return ``argmax_a Q(s, a)`` for a single observation."""
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        return int(self.forward(observation).argmax(dim=1).item())
