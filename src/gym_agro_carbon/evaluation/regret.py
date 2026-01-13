# src/gym_agro_carbon/evaluation/regret.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from gym_agro_carbon.models.context import ContextEncoder
from gym_agro_carbon.models.reward import RewardModel
from gym_agro_carbon.evaluation.oracle import OraclePolicy, mu_star_grid_for_observation


@dataclass(slots=True)
class RegretStep:
    """
    Per-step regret outputs (batch setting).
    """
    t: int
    regret_grid: np.ndarray      # (H,W) of mu*(x) - mu(a,x)
    regret: float                # scalar sum over grid
    mu_star_grid: np.ndarray     # (H,W) mu*(x)
    mu_a_grid: np.ndarray        # (H,W) mu(a,x)


@dataclass(slots=True)
class RegretTracker:
    """
    Tracks cumulative (pseudo-)regret in the discrete contextual batch setting.

    We compute the (pseudo-)regret using true mean rewards from the RewardModel:
        mu*(x)  = max_a mu(a, x)
        mu(a,x) = expected reward of chosen action under context x

    Per season (batch):
        r_t = sum_p [ mu*(x_t^p) - mu(a_t^p, x_t^p) ]
    Cumulative:
        R_T = sum_{t=1..T} r_t

    Note: This is NOT the realized regret based on sampled rewards; it is the
    theoretical pseudo-regret based on expected rewards.
    """

    encoder: ContextEncoder
    reward_model: RewardModel
    oracle: OraclePolicy

    cumulative_regret: float = 0.0
    step_regrets: Optional[list[float]] = None

    def __post_init__(self) -> None:
        if self.step_regrets is None:
            self.step_regrets = []

    def reset(self) -> None:
        self.cumulative_regret = 0.0
        self.step_regrets = []

    def compute_step(
        self,
        *,
        t: int,
        obs_context_ids: np.ndarray,
        actions_grid: np.ndarray,
    ) -> RegretStep:
        """
        Compute per-step pseudo-regret for a batch.

        Parameters
        ----------
        t:
            Current season index (for logging only).
        obs_context_ids:
            (H,W) matrix of context_id observed at decision time.
        actions_grid:
            (H,W) matrix of chosen actions (ints in [0..K-1]).

        Returns
        -------
        RegretStep containing regret grids and scalar regret.
        """
        obs_context_ids = np.asarray(obs_context_ids, dtype=np.int32)
        actions_grid = np.asarray(actions_grid, dtype=np.int32)

        if obs_context_ids.shape != actions_grid.shape:
            raise ValueError(
                f"obs_context_ids and actions_grid must have same shape. "
                f"Got {obs_context_ids.shape} vs {actions_grid.shape}."
            )
        if obs_context_ids.ndim != 2:
            raise ValueError(f"Expected 2D (H,W) arrays. Got {obs_context_ids.ndim}D.")

        # 1) mu*(x) from oracle lookup table
        mu_star_grid = mu_star_grid_for_observation(
            obs_context_ids=obs_context_ids,
            oracle=self.oracle,
        )  # float32 (H,W)

        # 2) mu(a,x) for chosen actions: compute deterministically via RewardModel
        # We decode contexts then evaluate mu_reward(action, s, tau)
        H, W = obs_context_ids.shape
        mu_a_grid = np.zeros((H, W), dtype=np.float32)

        # Small loops are OK for V1; can be vectorized later if needed.
        for i in range(H):
            for j in range(W):
                ctx_id = int(obs_context_ids[i, j])
                a = int(actions_grid[i, j])
                s, tau = self.encoder.from_id(ctx_id)
                mu_a_grid[i, j] = float(self.reward_model.mu_reward(a, s, tau))

        regret_grid = (mu_star_grid - mu_a_grid).astype(np.float32)
        regret = float(np.sum(regret_grid))

        return RegretStep(
            t=t,
            regret_grid=regret_grid,
            regret=regret,
            mu_star_grid=mu_star_grid,
            mu_a_grid=mu_a_grid,
        )

    def update(
        self,
        *,
        t: int,
        obs_context_ids: np.ndarray,
        actions_grid: np.ndarray,
    ) -> float:
        """
        Compute and accumulate pseudo-regret for a step.

        Returns the scalar regret for this step.
        """
        step = self.compute_step(t=t, obs_context_ids=obs_context_ids, actions_grid=actions_grid)
        self.cumulative_regret += step.regret
        self.step_regrets.append(step.regret)
        return step.regret


def batch_pseudo_regret(
    *,
    encoder: ContextEncoder,
    reward_model: RewardModel,
    oracle: OraclePolicy,
    obs_context_ids: np.ndarray,
    actions_grid: np.ndarray,
) -> float:
    """
    Convenience function for one-off pseudo-regret computation (scalar).
    """
    tracker = RegretTracker(encoder=encoder, reward_model=reward_model, oracle=oracle)
    step = tracker.compute_step(t=0, obs_context_ids=obs_context_ids, actions_grid=actions_grid)
    return step.regret
