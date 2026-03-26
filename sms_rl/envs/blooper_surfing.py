from __future__ import annotations

from collections import deque
import time
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
        self._last_path_progress = 0.0
        self._last_segment_index = 0
        self._steps = 0
        self._episode_start_s = 0.0
        self._finished_signal_count = 0
        self._failed_signal_count = 0
        self._path_segments = self._build_path_segments()
        self._path_start_point = (
            np.asarray(self.config.episode.path_waypoints[0], dtype=np.float64)
            if self.config.episode.path_waypoints
            else None
        )

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
        env_reset_start = time.perf_counter()
        env_reset_start_epoch = time.time()
        super().reset(seed=seed)
        del options

        state = self.driver.reset()
        state = self.driver.start_episode()
        processed = self._normalize_frame(state.frame)
        self._frame_stack.clear()
        for _ in range(self.config.observation.frame_stack):
            self._frame_stack.append(processed.copy())

        path_metrics = self._path_metrics_from_info(state.info)
        self._last_path_progress = path_metrics["path_progress"]
        self._last_segment_index = int(path_metrics["path_segment_index"])
        self._steps = 0
        self._episode_start_s = time.monotonic()
        self._finished_signal_count = 0
        self._failed_signal_count = 0
        info = dict(state.info)
        info["mission_finished"] = state.mission_finished
        info["mission_failed"] = state.mission_failed
        info["episode_steps"] = self._steps
        info["episode_elapsed_seconds"] = 0.0
        info["timeout_truncated"] = False
        info["raw_progress"] = state.progress
        info.update(path_metrics)
        info["progress"] = path_metrics["path_progress"]
        env_reset_end = time.perf_counter()
        env_reset_end_epoch = time.time()
        info["env_reset_started_epoch_s"] = env_reset_start_epoch
        info["env_reset_finished_epoch_s"] = env_reset_end_epoch
        info["env_reset_elapsed_s"] = env_reset_end - env_reset_start
        return self._stacked_observation(), info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        steering = SteeringAction(action)
        state = self.driver.step(steering, repeat=self.config.episode.action_repeat)
        self._frame_stack.append(self._normalize_frame(state.frame))
        self._steps += 1
        raw_finished = state.mission_finished
        raw_failed = state.mission_failed
        self._finished_signal_count = (
            self._finished_signal_count + 1 if raw_finished else 0
        )
        self._failed_signal_count = self._failed_signal_count + 1 if raw_failed else 0
        confirm_steps = max(1, self.config.episode.terminal_confirm_steps)
        confirmed_finished = self._finished_signal_count >= confirm_steps
        confirmed_failed = self._failed_signal_count >= confirm_steps
        debounced_state = StepState(
            frame=state.frame,
            progress=state.progress,
            mission_finished=confirmed_finished,
            mission_failed=confirmed_failed,
            info=state.info,
        )

        path_metrics = self._path_metrics_from_info(state.info)
        reward = self._compute_reward(debounced_state, path_metrics)
        terminated = confirmed_finished or confirmed_failed
        elapsed_s = time.monotonic() - self._episode_start_s
        timed_out = elapsed_s >= self.config.episode.max_episode_seconds
        truncated = (
            (self._steps >= self.config.episode.max_steps or timed_out)
            and not terminated
        )

        info = dict(state.info)
        info["mission_finished_raw"] = raw_finished
        info["mission_failed_raw"] = raw_failed
        info["mission_finished"] = confirmed_finished
        info["mission_failed"] = confirmed_failed
        info["mission_finished_confirm_count"] = self._finished_signal_count
        info["mission_failed_confirm_count"] = self._failed_signal_count
        info["episode_steps"] = self._steps
        info["episode_elapsed_seconds"] = elapsed_s
        info["timeout_truncated"] = bool(truncated and timed_out)
        info["raw_progress"] = state.progress
        info.update(path_metrics)
        info["progress"] = path_metrics["path_progress"]
        info["reward_components"] = self._reward_components(debounced_state, path_metrics)

        self._last_path_progress = path_metrics["path_progress"]
        self._last_segment_index = int(path_metrics["path_segment_index"])
        return self._stacked_observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self.driver.close()

    def _compute_reward(
        self,
        state: StepState,
        path_metrics: dict[str, float],
    ) -> float:
        components = self._reward_components(state, path_metrics)
        return float(sum(components.values()))

    def _reward_components(
        self,
        state: StepState,
        path_metrics: dict[str, float],
    ) -> dict[str, float]:
        progress_delta = max(0.0, path_metrics["path_progress"] - self._last_path_progress)
        components = {
            "progress": progress_delta * self.config.episode.progress_reward_scale,
            "path_distance": (
                -path_metrics["path_distance"]
                * self.config.episode.path_distance_penalty_scale
            ),
            "step_penalty": self.config.episode.step_penalty,
            "finish": (
                self.config.episode.finish_reward if state.mission_finished else 0.0
            ),
            "fail": self.config.episode.fail_reward if state.mission_failed else 0.0,
        }
        return components

    def _build_path_segments(
        self,
    ) -> list[tuple[int, np.ndarray, np.ndarray, float, float]]:
        waypoints = self.config.episode.path_waypoints
        if len(waypoints) < 2:
            return []

        segments: list[tuple[int, np.ndarray, np.ndarray, float, float]] = []
        cumulative = 0.0
        for idx in range(len(waypoints) - 1):
            start = np.asarray(waypoints[idx], dtype=np.float64)
            end = np.asarray(waypoints[idx + 1], dtype=np.float64)
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length <= 1e-6:
                continue
            segments.append((idx, start, delta, length, cumulative))
            cumulative += length
        return segments

    def _path_metrics_from_info(self, info: dict[str, Any]) -> dict[str, float]:
        x = info.get("position_x")
        z = info.get("position_z")
        if x is None or z is None or not self._path_segments:
            return {
                "path_progress": 0.0,
                "path_distance": 0.0,
                "path_segment_index": 0.0,
                "path_progress_raw": 0.0,
                "path_distance_to_start": 0.0,
                "path_progress_regression_clamped": 0.0,
                "path_respawn_detected": 0.0,
            }

        point = np.asarray([float(x), float(z)], dtype=np.float64)
        best_progress = 0.0
        best_distance = float("inf")
        best_segment_index = self._last_segment_index
        lookahead = max(0, self.config.episode.path_segment_lookahead)
        candidate_start = max(0, self._last_segment_index)
        candidate_end = min(len(self._path_segments) - 1, self._last_segment_index + lookahead)
        candidates = self._path_segments[candidate_start : candidate_end + 1]
        if not candidates:
            candidates = self._path_segments

        for segment_index, start, delta, length, cumulative in candidates:
            segment_direction = delta / length
            projection = float(np.dot(point - start, segment_direction))
            projection = min(max(projection, 0.0), length)
            closest = start + (segment_direction * projection)
            distance = float(np.linalg.norm(point - closest))
            progress = cumulative + projection
            if distance < best_distance:
                best_distance = distance
                best_progress = progress
                best_segment_index = segment_index

        progress_raw = best_progress
        regression_clamped = 0.0
        if best_progress + self.config.episode.path_backward_tolerance < self._last_path_progress:
            best_progress = self._last_path_progress
            best_segment_index = self._last_segment_index
            regression_clamped = 1.0

        distance_to_start = (
            float(np.linalg.norm(point - self._path_start_point))
            if self._path_start_point is not None
            else 0.0
        )
        respawn_detected = 0.0
        if (
            self._last_path_progress >= self.config.episode.path_respawn_progress_threshold
            and distance_to_start <= self.config.episode.path_respawn_start_distance_threshold
        ):
            best_progress = self._last_path_progress
            best_segment_index = self._last_segment_index
            respawn_detected = 1.0

        return {
            "path_progress": best_progress,
            "path_distance": 0.0 if best_distance == float("inf") else best_distance,
            "path_segment_index": float(best_segment_index),
            "path_progress_raw": progress_raw,
            "path_distance_to_start": distance_to_start,
            "path_progress_regression_clamped": regression_clamped,
            "path_respawn_detected": respawn_detected,
        }

    def _observation_shape(self) -> tuple[int, int, int]:
        channels_per_frame = 1 if self.config.observation.grayscale else 3
        return (
            self.config.observation.frame_stack * channels_per_frame,
            self.config.observation.height,
            self.config.observation.width,
        )

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        target_h = self.config.observation.height
        target_w = self.config.observation.width

        if self.config.observation.grayscale:
            gray = frame if frame.ndim == 2 else frame.mean(axis=2).astype(np.uint8)
            resized = self._resize_nearest(gray, target_h, target_w)
            return resized.astype(np.uint8, copy=False)

        color = frame
        if color.ndim == 2:
            color = np.repeat(color[:, :, None], 3, axis=2)
        resized = self._resize_nearest(color, target_h, target_w)
        return np.transpose(resized.astype(np.uint8, copy=False), (2, 0, 1))

    @staticmethod
    def _resize_nearest(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        src_h, src_w = frame.shape[:2]
        if src_h == target_h and src_w == target_w:
            return frame
        y_idx = np.linspace(0, src_h - 1, target_h).astype(np.int32)
        x_idx = np.linspace(0, src_w - 1, target_w).astype(np.int32)
        return frame[y_idx][:, x_idx]

    def _stacked_observation(self) -> np.ndarray:
        frames = list(self._frame_stack)
        if self.config.observation.grayscale:
            return np.stack(frames, axis=0)
        return np.concatenate(frames, axis=0)
