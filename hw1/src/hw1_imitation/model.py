"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layer_dims = (state_dim, *hidden_dims, chunk_size * action_dim)
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(layer_dims[:-1], layer_dims[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        predicted = self.layers(state)
        target = action_chunk.reshape(state.size(0), -1)
        return nn.functional.mse_loss(predicted, target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            predicted = self.layers(state)
            return predicted.view(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        chunk_dim = chunk_size * action_dim
        layer_dims = (state_dim + chunk_dim + 1, *hidden_dims, chunk_dim)
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(layer_dims[:-1], layer_dims[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.size(0)
        actions_tau1 = action_chunk.reshape(batch_size, -1)
        actions_tau0 = torch.randn_like(actions_tau1)
        tau = torch.rand(
            batch_size,
            1,
            device=state.device,
            dtype=state.dtype,
        )

        actions_tau = tau * actions_tau1 + (1.0 - tau) * actions_tau0
        inputs = torch.cat([state, actions_tau, tau], dim=1)
        predicted_velocities = self.layers(inputs)
        target_velocities = actions_tau1 - actions_tau0
        return nn.functional.mse_loss(predicted_velocities, target_velocities)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        batch_size = state.size(0)
        chunk_dim = self.chunk_size * self.action_dim
        with torch.no_grad():
            actions_tau = torch.randn(
                batch_size,
                chunk_dim,
                device=state.device,
                dtype=state.dtype,
            )
            step_size = 1.0 / num_steps
            for step in range(num_steps):
                tau = actions_tau.new_full((batch_size, 1), step * step_size)
                inputs = torch.cat([state, actions_tau, tau], dim=1)
                velocity = self.layers(inputs)
                actions_tau = actions_tau + step_size * velocity

        return actions_tau.reshape(batch_size, self.chunk_size, self.action_dim)

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
