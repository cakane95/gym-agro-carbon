# src/gym_agro_carbon/bandits/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from gym_agro_carbon.models.context import ContextEncoder


class BatchContextualBandit(ABC):
    """
    Abstract base class for a Batch Contextual Bandit agent.

    The agent operates on a grid (batch setting):
    - Input observation: (H, W) context_id matrix
    - Output decision:   (H, W) action matrix
    - Feedback:          (H, W) local reward matrix (per parcel)
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        num_actions: int,
        seed: Optional[int] = None,
        name: str = "BaseAgent",
    ) -> None:
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive. Got {num_actions}.")
        self.encoder = context_encoder
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def reset(self) -> None:
        """Reset the internal state of the agent (counters, posteriors, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def select_actions(self, obs_context_ids: np.ndarray) -> np.ndarray:
        """
        Select actions for the entire grid.

        Parameters
        ----------
        obs_context_ids : np.ndarray
            (H, W) matrix of discrete context IDs.

        Returns
        -------
        actions : np.ndarray
            (H, W) matrix of selected action indices (int32 in [0..K-1]).
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        obs_context_ids: np.ndarray,
        actions: np.ndarray,
        reward_grid: np.ndarray,
    ) -> None:
        """
        Update internal belief/policy using local feedback.

        Parameters
        ----------
        obs_context_ids : np.ndarray
            (H, W) matrix of contexts used for the decision.
        actions : np.ndarray
            (H, W) matrix of actions chosen by the agent.
        reward_grid : np.ndarray
            (H, W) matrix of local rewards returned by the environment.
        """
        raise NotImplementedError

    # Optional helper for cheap validation (keeps subclasses clean)
    def _validate_batch_inputs(
        self,
        obs_context_ids: np.ndarray,
        actions: np.ndarray,
        reward_grid: np.ndarray,
    ) -> None:
        if obs_context_ids.shape != actions.shape or obs_context_ids.shape != reward_grid.shape:
            raise ValueError(
                "obs_context_ids, actions, and reward_grid must have the same shape. "
                f"Got obs={obs_context_ids.shape}, actions={actions.shape}, rewards={reward_grid.shape}."
            )
        if obs_context_ids.ndim != 2:
            raise ValueError(f"Expected 2D (H,W) arrays. Got {obs_context_ids.ndim}D.")
        if actions.min() < 0 or actions.max() >= self.num_actions:
            raise ValueError(
                f"actions must be in [0..{self.num_actions - 1}]. "
                f"Got [{int(actions.min())}..{int(actions.max())}]."
            )


class RandomBandit(BatchContextualBandit):
    """
    Uniform random policy (sanity check baseline).
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        num_actions: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(context_encoder, num_actions, seed, name="Random")

    def reset(self) -> None:
        # Nothing to reset for a random policy.
        return

    def select_actions(self, obs_context_ids: np.ndarray) -> np.ndarray:
        return self.rng.integers(
            low=0, high=self.num_actions, size=obs_context_ids.shape, dtype=np.int32
        )

    def update(
        self,
        obs_context_ids: np.ndarray,
        actions: np.ndarray,
        reward_grid: np.ndarray,
    ) -> None:
        # Random agent does not learn.
        return
