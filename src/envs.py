"""Vectorized Gymnasium environment construction."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np


def make_env(env_id: str, seed: int, rank: int) -> Callable[[], gym.Env]:
    """Return a thunk that creates one independently seeded environment."""

    def thunk() -> gym.Env:
        env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return thunk


def create_vector_env(env_id: str, num_envs: int, seed: int) -> gym.vector.AsyncVectorEnv:
    """Create parallel environments with continuous actions clipped by Gymnasium."""

    env = gym.vector.AsyncVectorEnv(
        [make_env(env_id, seed, rank) for rank in range(num_envs)],
        shared_memory=True,
        copy=True,
    )
    if not isinstance(env.single_action_space, gym.spaces.Box):
        raise TypeError("PPO requires a continuous Box action space")
    return env


def clip_actions(actions: np.ndarray, action_space: gym.spaces.Box) -> np.ndarray:
    """Keep sampled continuous actions within the environment's bounds."""

    return np.clip(actions, action_space.low, action_space.high).astype(np.float32)