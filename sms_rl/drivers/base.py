from __future__ import annotations

from enum import IntEnum
from typing import Protocol

from sms_rl.types import StepState


class SteeringAction(IntEnum):
    LEFT = 0
    NEUTRAL = 1
    RIGHT = 2
    JUMP = 3


class BlooperDriver(Protocol):
    """Backend contract for one Blooper Surfing episode loop.

    A concrete driver can use keyboard injection, virtual controller input,
    Dolphin memory reads, or any combination that produces consistent state.
    """

    def reset(self) -> StepState:
        """Restore the fixed mission start state and return the first frame."""

    def start_episode(self) -> StepState:
        """Run any deterministic pre-episode setup and return the first controllable state."""

    def step(self, action: SteeringAction, repeat: int) -> StepState:
        """Apply an action for `repeat` emulator frames and return the latest state."""

    def close(self) -> None:
        """Release any capture or emulator resources."""
