# src/gym_agro_carbon/evaluation/oracle.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gym_agro_carbon.models.context import ContextEncoder
from gym_agro_carbon.models.reward import RewardModel


@dataclass(slots=True)
class OraclePolicy:
    """
    Oracle policy for the discrete contextual setting.

    The oracle knows the true mean reward mu(a, x) induced by the RewardModel
    (i.e., it can query RewardModel.mu_reward(action, s, tau)).

    It returns:
    - best_action_by_context[context_id] : optimal action for that context
    - mu_star_by_context[context_id]     : optimal mean reward for that context
    """
    best_action_by_context: np.ndarray  # shape (num_contexts,), dtype int32
    mu_star_by_context: np.ndarray      # shape (num_contexts,), dtype float32


def build_oracle_policy(
    *,
    encoder: ContextEncoder,
    reward_model: RewardModel,
    num_actions: int,
) -> OraclePolicy:
    """
    Build a full lookup-table oracle policy over all discrete contexts.
    """
    if num_actions <= 0:
        raise ValueError(f"num_actions must be positive. Got {num_actions}.")

    num_contexts = encoder.spec.num_contexts
    best_actions = np.zeros((num_contexts,), dtype=np.int32)
    mu_star = np.zeros((num_contexts,), dtype=np.float32)

    for ctx_id in range(num_contexts):
        s, tau = encoder.from_id(ctx_id)

        best_a = 0
        best_mu = reward_model.mu_reward(0, s, tau)

        for a in range(1, num_actions):
            mu = reward_model.mu_reward(a, s, tau)
            if mu > best_mu:
                best_mu = mu
                best_a = a

        best_actions[ctx_id] = best_a
        mu_star[ctx_id] = float(best_mu)

    return OraclePolicy(best_action_by_context=best_actions, mu_star_by_context=mu_star)


def oracle_actions_for_observation(
    *,
    obs_context_ids: np.ndarray,
    oracle: OraclePolicy,
) -> np.ndarray:
    """
    Map an (H,W) matrix of context_ids to an (H,W) matrix of oracle actions.
    """
    obs_context_ids = np.asarray(obs_context_ids, dtype=np.int32)
    if obs_context_ids.ndim != 2:
        raise ValueError(f"obs_context_ids must be 2D (H,W). Got shape {obs_context_ids.shape}.")

    if obs_context_ids.min() < 0 or obs_context_ids.max() >= oracle.best_action_by_context.shape[0]:
        raise ValueError(
            "obs_context_ids contains invalid context ids. "
            f"Valid range is [0..{oracle.best_action_by_context.shape[0]-1}], "
            f"got [{int(obs_context_ids.min())}..{int(obs_context_ids.max())}]."
        )

    return oracle.best_action_by_context[obs_context_ids].astype(np.int32)


def mu_star_grid_for_observation(
    *,
    obs_context_ids: np.ndarray,
    oracle: OraclePolicy,
) -> np.ndarray:
    """
    Return the oracle optimal mean reward mu*(x) for each parcel context in obs.
    """
    obs_context_ids = np.asarray(obs_context_ids, dtype=np.int32)
    if obs_context_ids.ndim != 2:
        raise ValueError(f"obs_context_ids must be 2D (H,W). Got shape {obs_context_ids.shape}.")

    if obs_context_ids.min() < 0 or obs_context_ids.max() >= oracle.mu_star_by_context.shape[0]:
        raise ValueError(
            "obs_context_ids contains invalid context ids. "
            f"Valid range is [0..{oracle.mu_star_by_context.shape[0]-1}], "
            f"got [{int(obs_context_ids.min())}..{int(obs_context_ids.max())}]."
        )

    return oracle.mu_star_by_context[obs_context_ids].astype(np.float32)
