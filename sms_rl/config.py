from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(slots=True)
class ObservationConfig:
    width: int = 96
    height: int = 72
    grayscale: bool = True
    frame_stack: int = 4


@dataclass(slots=True)
class EpisodeConfig:
    max_steps: int = 900
    max_episode_seconds: float = 45.0
    action_repeat: int = 4
    terminal_confirm_steps: int = 2
    step_penalty: float = -0.01
    finish_reward: float = 25.0
    fail_reward: float = -25.0
    progress_reward_scale: float = 1.0
    path_waypoints: Tuple[tuple[float, float], ...] = ()


@dataclass(slots=True)
class EnvConfig:
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
