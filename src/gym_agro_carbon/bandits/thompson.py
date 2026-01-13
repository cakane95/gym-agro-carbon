# src/gym_agro_carbon/bandits/thompson.py
from __future__ import annotations

from typing import Optional

import numpy as np

from gym_agro_carbon.bandits.base import BatchContextualBandit
from gym_agro_carbon.models.context import ContextEncoder


class ThompsonSamplingBandit(BatchContextualBandit):
    """
    Contextual Thompson Sampling for batch decision-making (discrete contexts).

    V1 assumptions:
    - Reward r(a, x) is modeled as Gaussian.
    - Prior for each (context x, action a): Normal(mu0, sigma0^2).
    - Observation noise variance is fixed and set to 1.0 by default:
        r ~ Normal(theta, obs_noise_var)
      This is a modeling choice for V1 (can be tuned later).

    For each (x,a), we maintain a Normal posterior:
      theta(x,a) | data ~ Normal(mean[x,a], var[x,a])

    Decision rule (per cell):
      - For each grid cell, sample theta ~ Normal(mean, var) for its context
      - Choose argmax_a theta

    Update rule (conjugate Normal-Normal with known variance):
      precision_post = precision_prior + n / obs_noise_var
      mean_post = var_post * (mean_prior*precision_prior + sum_rewards/obs_noise_var)
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        num_actions: int,
        seed: Optional[int] = None,
        mu0: float = 0.0,
        sigma0: float = 1.0,
        obs_noise_var: float = 1.0,
        name: str = "ThompsonSampling",
    ) -> None:
        super().__init__(context_encoder, num_actions, seed, name)

        if sigma0 <= 0:
            raise ValueError("sigma0 must be > 0")
        if obs_noise_var <= 0:
            raise ValueError("obs_noise_var must be > 0")

        self.mu0 = float(mu0)
        self.sigma0 = float(sigma0)
        self.obs_noise_var = float(obs_noise_var)

        self.means: np.ndarray = np.empty(0, dtype=np.float32)   # (C,K)
        self.vars: np.ndarray = np.empty(0, dtype=np.float32)    # (C,K)
        self.counts: np.ndarray = np.empty(0, dtype=np.int32)    # (C,K)

        self.reset()

    def reset(self) -> None:
        num_contexts = self.encoder.spec.num_contexts

        self.means = np.full(
            (num_contexts, self.num_actions),
            self.mu0,
            dtype=np.float32,
        )
        self.vars = np.full(
            (num_contexts, self.num_actions),
            self.sigma0 ** 2,
            dtype=np.float32,
        )
        self.counts = np.zeros(
            (num_contexts, self.num_actions),
            dtype=np.int32,
        )

    def select_actions(self, obs_context_ids: np.ndarray) -> np.ndarray:
        obs_context_ids = np.asarray(obs_context_ids, dtype=np.int32)
        if obs_context_ids.ndim != 2:
            raise ValueError(f"Expected 2D grid, got {obs_context_ids.shape}")

        # Sample per cell (H,W,K) to allow diversity even for repeated contexts
        grid_means = self.means[obs_context_ids]  # (H,W,K)
        grid_vars = self.vars[obs_context_ids]    # (H,W,K)
        grid_stds = np.sqrt(grid_vars)

        grid_theta = self.rng.normal(loc=grid_means, scale=grid_stds)  # (H,W,K)

        actions = np.argmax(grid_theta, axis=-1).astype(np.int32)      # (H,W)
        return actions

    def update(
        self,
        obs_context_ids: np.ndarray,
        actions: np.ndarray,
        reward_grid: np.ndarray,
    ) -> None:
        """
        Vectorized posterior update using aggregated sufficient statistics per (context, action).
        """
        self._validate_batch_inputs(obs_context_ids, actions, reward_grid)

        flat_ctx = np.asarray(obs_context_ids, dtype=np.int32).ravel()
        flat_act = np.asarray(actions, dtype=np.int32).ravel()
        flat_rew = np.asarray(reward_grid, dtype=np.float32).ravel()

        # Aggregate counts and sum of rewards for each (context, action)
        flat_indices = flat_ctx * self.num_actions + flat_act
        num_params = self.means.size  # C*K

        batch_counts_1d = np.bincount(flat_indices, minlength=num_params).astype(np.float32)
        batch_sums_1d = np.bincount(flat_indices, weights=flat_rew, minlength=num_params).astype(np.float32)

        batch_counts = batch_counts_1d.reshape(self.means.shape)  # (C,K)
        batch_sums = batch_sums_1d.reshape(self.means.shape)      # (C,K)

        mask = batch_counts > 0
        if not np.any(mask):
            return

        # Conjugate Normal-Normal update with known observation variance
        prev_precision = 1.0 / self.vars  # (C,K)

        # precision_post = precision_prior + n / obs_noise_var
        precision_post = prev_precision.copy()
        precision_post[mask] = prev_precision[mask] + (batch_counts[mask] / self.obs_noise_var)

        var_post = self.vars.copy()
        var_post[mask] = 1.0 / precision_post[mask]

        # mean_post = var_post * (mean_prior*precision_prior + sum_rewards/obs_noise_var)
        mean_post = self.means.copy()
        mean_post[mask] = var_post[mask] * (
            (self.means[mask] * prev_precision[mask]) + (batch_sums[mask] / self.obs_noise_var)
        )

        self.means = mean_post.astype(np.float32)
        self.vars = var_post.astype(np.float32)
        self.counts += batch_counts.astype(np.int32)