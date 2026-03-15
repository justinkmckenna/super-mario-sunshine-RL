from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sms_rl.config import ObservationConfig
from sms_rl.drivers.base import SteeringAction
from sms_rl.types import StepState


@dataclass(slots=True)
class MockTrackConfig:
    track_length: float = 100.0
    lateral_limit: float = 1.0
    forward_speed: float = 1.0
    steering_delta: float = 0.12


class MockBlooperDriver:
    """Deterministic stand-in used to validate the env loop before Dolphin wiring."""

    def __init__(
        self,
        observation: ObservationConfig,
        track: MockTrackConfig | None = None,
    ) -> None:
        self._observation = observation
        self._track = track or MockTrackConfig()
        self._progress = 0.0
        self._lateral_offset = 0.0
        self._step_count = 0

    def reset(self) -> StepState:
        self._progress = 0.0
        self._lateral_offset = 0.0
        self._step_count = 0
        return self._state()

    def start_episode(self) -> StepState:
        return self._state()

    def step(self, action: SteeringAction, repeat: int) -> StepState:
        is_jump = action == SteeringAction.JUMP
        steer = {
            SteeringAction.LEFT: -1.0,
            SteeringAction.NEUTRAL: 0.0,
            SteeringAction.RIGHT: 1.0,
            SteeringAction.JUMP: 0.0,
        }[action]

        for _ in range(repeat):
            self._step_count += 1
            self._progress += self._track.forward_speed
            self._lateral_offset += steer * self._track.steering_delta
            if is_jump:
                jump_impulse = 0.24 if (self._step_count % 2 == 0) else -0.24
                self._lateral_offset += jump_impulse
            self._lateral_offset *= 0.92
            if abs(self._lateral_offset) > self._track.lateral_limit:
                break

        return self._state()

    def close(self) -> None:
        return None

    def _state(self) -> StepState:
        finished = self._progress >= self._track.track_length
        failed = abs(self._lateral_offset) > self._track.lateral_limit
        frame = self._render_frame()
        info = {
            "mock_progress": self._progress,
            "mock_lateral_offset": self._lateral_offset,
            "mock_step_count": self._step_count,
        }
        return StepState(
            frame=frame,
            progress=min(self._progress, self._track.track_length),
            mission_finished=finished,
            mission_failed=failed,
            info=info,
        )

    def _render_frame(self) -> np.ndarray:
        height = self._observation.height
        width = self._observation.width
        frame = np.zeros((height, width), dtype=np.uint8)

        center_x = width // 2
        lane_half_width = max(8, width // 6)
        frame[:, center_x - lane_half_width : center_x + lane_half_width] = 40

        progress_ratio = min(1.0, self._progress / self._track.track_length)
        horizon_row = int((1.0 - progress_ratio) * (height - 1))
        frame[horizon_row:, center_x - 1 : center_x + 1] = 100

        blooper_x = int(center_x + self._lateral_offset * lane_half_width)
        blooper_x = max(2, min(width - 3, blooper_x))
        frame[height - 8 : height - 3, blooper_x - 2 : blooper_x + 2] = 255

        if not self._observation.grayscale:
            frame = np.repeat(frame[:, :, None], 3, axis=2)

        return frame
