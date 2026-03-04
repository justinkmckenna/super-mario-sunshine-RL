from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sms_rl.config import EnvConfig
from sms_rl.drivers.base import BlooperDriver, SteeringAction
from sms_rl.drivers.mock import MockBlooperDriver
from sms_rl.types import StepState


class BlooperSurfingEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        driver: BlooperDriver | None = None,
    ) -> None:
        self.config = config or EnvConfig()
        self.driver = driver or MockBlooperDriver(self.config.observation)
        self._frame_stack: deque[np.ndarray] = deque(
            maxlen=self.config.observation.frame_stack
        )
        self._last_progress = 0.0
        self._steps = 0

        obs_shape = self._observation_shape()
        self.action_space = spaces.Discrete(len(SteeringAction))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        state = self.driver.reset()
        processed = self._normalize_frame(state.frame)
        self._frame_stack.clear()
        for _ in range(self.config.observation.frame_stack):
            self._frame_stack.append(processed.copy())

        self._last_progress = state.progress
        self._steps = 0
        return self._stacked_observation(), dict(state.info)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        steering = SteeringAction(action)
        state = self.driver.step(steering, repeat=self.config.episode.action_repeat)
        self._frame_stack.append(self._normalize_frame(state.frame))
        self._steps += 1

        reward = self._compute_reward(state)
        terminated = state.mission_finished or state.mission_failed
        truncated = self._steps >= self.config.episode.max_steps and not terminated

        info = dict(state.info)
        info["mission_finished"] = state.mission_finished
        info["mission_failed"] = state.mission_failed
        info["episode_steps"] = self._steps
        info["progress"] = state.progress
        info["reward_components"] = self._reward_components(state)

        self._last_progress = state.progress
        return self._stacked_observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self.driver.close()

    def _compute_reward(self, state: StepState) -> float:
        components = self._reward_components(state)
        return float(sum(components.values()))

    def _reward_components(self, state: StepState) -> dict[str, float]:
        progress_delta = max(0.0, state.progress - self._last_progress)
        components = {
            "progress": progress_delta * self.config.episode.progress_reward_scale,
            "step_penalty": self.config.episode.step_penalty,
            "finish": (
                self.config.episode.finish_reward if state.mission_finished else 0.0
            ),
            "fail": self.config.episode.fail_reward if state.mission_failed else 0.0,
        }
        return components

    def _observation_shape(self) -> tuple[int, int, int]:
        channels_per_frame = 1 if self.config.observation.grayscale else 3
        return (
            self.config.observation.frame_stack * channels_per_frame,
            self.config.observation.height,
            self.config.observation.width,
        )

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.config.observation.grayscale and frame.ndim == 2:
            return frame
        if self.config.observation.grayscale and frame.ndim == 3:
            return frame.mean(axis=2).astype(np.uint8)
        if not self.config.observation.grayscale and frame.ndim == 2:
            return np.repeat(frame[None, :, :], 3, axis=0)
        return np.transpose(frame, (2, 0, 1))

    def _stacked_observation(self) -> np.ndarray:
        frames = list(self._frame_stack)
        if self.config.observation.grayscale:
            return np.stack(frames, axis=0)
        return np.concatenate(frames, axis=0)
