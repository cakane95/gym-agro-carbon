from __future__ import annotations
from typing import Optional
import numpy as np

from gym_agro_carbon.bandits.base import BatchContextualBandit
from gym_agro_carbon.models.context import ContextEncoder

class MetaExploreThenCommitBandit(BatchContextualBandit):
    """
    Explore-Then-Commit (ETC) Meta Agent.
    
    Explores each action once for the entire grid during the first K seasons,
    then commits to the best performing action for the rest of the horizon.
    This agent is context-agnostic (Meta level).
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        num_actions: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(context_encoder, num_actions, seed, name="ETC-Meta")
        
        # Internal state
        self.total_rewards_per_action = np.zeros(num_actions)
        self.current_season = 0
        self.best_action: Optional[int] = None

    def reset(self) -> None:
        """Resets the bandit state for a new episode."""
        self.total_rewards_per_action.fill(0.0)
        self.current_season = 0
        self.best_action = None

    def select_actions(self, obs_context_ids: np.ndarray) -> np.ndarray:
        """
        Decision rule: 
        - If t < K: Explore the current season's action.
        - If t >= K: Play the best action identified during exploration.
        """
        # Phase 1: Exploration
        if self.current_season < self.num_actions:
            action_to_play = self.current_season
        # Phase 2: Commitment
        else:
            if self.best_action is None:
                self.best_action = int(np.argmax(self.total_rewards_per_action))
            action_to_play = self.best_action

        return np.full(obs_context_ids.shape, action_to_play, dtype=np.int32)

    def update(
        self,
        obs_context_ids: np.ndarray,
        actions: np.ndarray,
        reward_grid: np.ndarray,
    ) -> None:
        """
        Updates the meta-reward statistics.
        Since it's a Meta agent, it aggregates all parcel rewards.
        """
        self._validate_batch_inputs(obs_context_ids, actions, reward_grid)
        
        # We only care about updates during the exploration phase (first K seasons)
        if self.current_season < self.num_actions:
            # Aggregate reward for the action played this season
            # In Meta, we treat the whole batch as one big feedback
            action_played = actions[0, 0] # All cells played the same action
            self.total_rewards_per_action[action_played] += np.mean(reward_grid)
            
        self.current_season += 1

class ContextualExploreThenCommitBandit(BatchContextualBandit):
    """
    Contextual Explore-Then-Commit (ETC) Agent.
    
    For each discrete context (soil type x tree age), the agent explores 
    all available actions once. After the exploration phase for a specific 
    context is complete, it commits to the best performing action for that context.
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        num_actions: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(context_encoder, num_actions, seed, name="ETC-Context")
        
        self.num_contexts = self.encoder.spec.num_contexts
        
        # Stats per (context, action)
        self.sum_rewards = np.zeros((self.num_contexts, num_actions))
        self.counts = np.zeros((self.num_contexts, num_actions), dtype=np.int32)
        
        # Optimization: Cache for the best action per context
        self.best_actions = np.zeros(self.num_contexts, dtype=np.int32)

    def reset(self) -> None:
        """Resets counters and rewards for a new episode."""
        self.sum_rewards.fill(0.0)
        self.counts.fill(0)
        self.best_actions.fill(0)

    def select_actions(self, obs_context_ids: np.ndarray) -> np.ndarray:
        """
        Decision rule per cell:
        - If context x has unexplored actions: pick the next unexplored action.
        - If context x is fully explored: pick the best action based on mean reward.
        """
        # (H, W) matrix to store choices
        actions = np.zeros_like(obs_context_ids, dtype=np.int32)
        
        # Flatten for vectorized check
        flat_contexts = obs_context_ids.ravel()
        flat_actions = np.zeros_like(flat_contexts)

        for i, ctx in enumerate(flat_contexts):
            # Find which actions have been tested for this context
            tested_count = np.sum(self.counts[ctx] > 0)
            
            if tested_count < self.num_actions:
                # Still exploring: pick the next action index
                # (Simple deterministic exploration: 0, then 1, then 2...)
                flat_actions[i] = tested_count
            else:
                # Commit phase: use the cached best action
                flat_actions[i] = self.best_actions[ctx]
                
        return flat_actions.reshape(obs_context_ids.shape)

    def update(
        self,
        obs_context_ids: np.ndarray,
        actions: np.ndarray,
        reward_grid: np.ndarray,
    ) -> None:
        """
        Update rewards and counts per context-action pair.
        """
        self._validate_batch_inputs(obs_context_ids, actions, reward_grid)
        
        flat_ctx = obs_context_ids.ravel()
        flat_act = actions.ravel()
        flat_rew = reward_grid.ravel()

        # Update statistics using in-place addition at indices
        np.add.at(self.counts, (flat_ctx, flat_act), 1)
        np.add.at(self.sum_rewards, (flat_ctx, flat_act), flat_rew)

        # Update the best action cache for contexts that finished exploration
        # We only re-calculate for contexts present in this batch for efficiency
        unique_contexts = np.unique(flat_ctx)
        for ctx in unique_contexts:
            if np.all(self.counts[ctx] > 0):
                means = self.sum_rewards[ctx] / self.counts[ctx]
                self.best_actions[ctx] = np.argmax(means)