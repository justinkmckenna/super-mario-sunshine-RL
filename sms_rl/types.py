from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Frame = NDArray[np.uint8]


@dataclass(slots=True)
class StepState:
    frame: Frame
    progress: float
    mission_finished: bool
    mission_failed: bool
    info: dict[str, object]
