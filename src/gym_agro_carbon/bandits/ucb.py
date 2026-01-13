# src/gym_agro_carbon/bandits/ucb.py
from __future__ import annotations

from typing import Optional

import numpy as np

from gym_agro_carbon.bandits.base import BatchContextualBandit
from gym_agro_carbon.models.context import ContextEncoder


class UCBBandit(BatchContextualBandit):
    """
    Upper Confidence Bound agent for Discrete Contextual Bandits (batch setting).

    For each discrete context x and action a, we track:
      - N(x,a): pull count
      - S(x,a): sum of observed local rewards
      - N(x):   total visits of context x

    Decision rule (per-context UCB):
      a = argmax_a [ Q(x,a) + c * sqrt( log(max(1, N(x))) / N(x,a) ) ]

    Where:
      Q(x,a) = S(x,a) / N(x,a)
      Untried actions (N(x,a)=0) receive +inf bonus (forced exploration).
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        num_actions: int,
        seed: Optional[int] = None,
        exploration_c: float = 1.0,
        name: str = "UCB",
    ) -> None:
        super().__init__(context_encoder, num_actions, seed, name)
        if exploration_c <= 0:
            raise ValueError(f"exploration_c must be > 0. Got {exploration_c}.")
        self.exploration_c = float(exploration_c)

        self.counts: np.ndarray = np.empty(0, dtype=np.int32)         # (C,K)
        self.sums: np.ndarray = np.empty(0, dtype=np.float32)         # (C,K)
        self.context_counts: np.ndarray = np.empty(0, dtype=np.int32) # (C,)

        self.reset()

    def reset(self) -> None:
        num_contexts = self.encoder.spec.num_contexts
        self.counts = np.zeros((num_contexts, self.num_actions), dtype=np.int32)
        self.sums = np.zeros((num_contexts, self.num_actions), dtype=np.float32)
        self.context_counts = np.zeros((num_contexts,), dtype=np.int32)

    def select_actions(self, obs_context_ids: np.ndarray) -> np.ndarray:
        obs_context_ids = np.asarray(obs_context_ids, dtype=np.int32)
        if obs_context_ids.ndim != 2:
            raise ValueError(f"obs_context_ids must be 2D (H,W). Got {obs_context_ids.shape}.")

        # Empirical means Q(x,a)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = self.sums / self.counts
            means[self.counts == 0] = 0.0

        # Per-context exploration term log(max(1, N(x)))
        Nx = self.context_counts.astype(np.float32)[:, None]  # (C,1)
        log_Nx = np.log(np.maximum(1.0, Nx))                  # (C,1)

        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = self.exploration_c * np.sqrt(log_Nx / self.counts)
            bonus[self.counts == 0] = np.inf

        ucb_scores = means + bonus  # (C,K)

        # Map scores to grid contexts -> (H,W,K) then argmax
        grid_scores = ucb_scores[obs_context_ids]
        actions = np.argmax(grid_scores, axis=-1).astype(np.int32)
        return actions

    def update(
        self,
        obs_context_ids: np.ndarray,
        actions: np.ndarray,
        reward_grid: np.ndarray,
    ) -> None:
        self._validate_batch_inputs(obs_context_ids, actions, reward_grid)

        flat_ctx = np.asarray(obs_context_ids, dtype=np.int32).ravel()
        flat_act = np.asarray(actions, dtype=np.int32).ravel()
        flat_rew = np.asarray(reward_grid, dtype=np.float32).ravel()

        np.add.at(self.counts, (flat_ctx, flat_act), 1)
        np.add.at(self.sums, (flat_ctx, flat_act), flat_rew)
        np.add.at(self.context_counts, flat_ctx, 1)