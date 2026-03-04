from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Policy(Protocol):
    def act(self, observation: np.ndarray) -> int:
        """Return a discrete action for the current observation."""


class ConstantPolicy:
    def __init__(self, action: int) -> None:
        self._action = action

    def act(self, observation: np.ndarray) -> int:
        del observation
        return self._action


class RandomPolicy:
    def __init__(self, action_count: int, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._action_count = action_count

    def act(self, observation: np.ndarray) -> int:
        del observation
        return int(self._rng.integers(0, self._action_count))


class CenteringPolicy:
    """Tiny heuristic baseline for the mock driver.

    It reads the brightest block near the bottom of the frame and steers back
    toward the center lane. This is not intended for Dolphin frames.
    """

    def act(self, observation: np.ndarray) -> int:
        latest = observation[-1]
        bottom = latest[-12:]
        x_indices = np.where(bottom >= 250)[1]
        if x_indices.size == 0:
            return 1

        blooper_center = float(x_indices.mean())
        frame_center = latest.shape[1] / 2.0
        delta = blooper_center - frame_center
        if delta < -3:
            return 2
        if delta > 3:
            return 0
        return 1


@dataclass(slots=True)
class EpisodeResult:
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    info: dict[str, object]


def run_episode(env, policy: Policy) -> EpisodeResult:
    observation, info = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    return EpisodeResult(
        total_reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )
