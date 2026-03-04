from __future__ import annotations

import ctypes
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from sms_rl.config import ObservationConfig
from sms_rl.drivers.base import SteeringAction
from sms_rl.types import StepState


ScalarType = Literal["byte", "word", "float", "double"]


class DolphinDriverError(RuntimeError):
    """Raised when the Windows Dolphin driver cannot complete an operation."""


@dataclass(slots=True)
class DolphinLaunchConfig:
    dolphin_path: Path
    game_path: Path
    save_state_path: Path | None = None
    batch_mode: bool = False
    user_path: Path | None = None
    render_to_main: bool = False
    window_title_contains: str = "Dolphin"
    launch_timeout_s: float = 20.0
    stable_window_time_s: float = 1.0


@dataclass(slots=True)
class CaptureConfig:
    region: tuple[int, int, int, int] | None = None
    output_color: Literal["gray", "rgb"] = "gray"
    target_fps: int = 60
    warmup_frames: int = 10


@dataclass(slots=True)
class MemoryValueSpec:
    base_address: int
    value_type: ScalarType = "float"
    pointer_offsets: tuple[int, ...] = ()


@dataclass(slots=True)
class MemoryFlagSpec(MemoryValueSpec):
    expected_value: int | float = 1


@dataclass(slots=True)
class MemoryBindings:
    progress: MemoryValueSpec | None = None
    mission_finished: MemoryFlagSpec | None = None
    mission_failed: MemoryFlagSpec | None = None


@dataclass(slots=True)
class DolphinDriverConfig:
    launch: DolphinLaunchConfig
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    memory: MemoryBindings = field(default_factory=MemoryBindings)
    restart_on_reset: bool = True
    neutral_deadzone: float = 0.2
    left_stick_magnitude: float = 0.75
    post_launch_delay_s: float = 2.0
    post_reset_delay_s: float = 0.5
    step_sleep_s: float = 0.0


class DolphinWindowsDriver:
    """Windows Dolphin driver backed by DXcam, vgamepad, and memory-engine.

    This driver is intentionally generic about game-state addresses. It can run
    only after the project supplies working addresses for progress, mission
    success, and mission failure.
    """

    def __init__(self, config: DolphinDriverConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._camera = None
        self._window_handle: int | None = None
        self._capture_region: tuple[int, int, int, int] | None = None
        self._gamepad = self._create_gamepad()
        self._memory = self._load_memory_engine()
        self._dxcam = None

    def reset(self) -> StepState:
        if self.config.restart_on_reset or self._process is None:
            self._restart_dolphin()

        self._center_steering()
        time.sleep(self.config.post_reset_delay_s)
        return self._read_state()

    def step(self, action: SteeringAction, repeat: int) -> StepState:
        self._ensure_runtime_ready()
        self._apply_steering(action)
        if self.config.step_sleep_s > 0:
            time.sleep(self.config.step_sleep_s * repeat)
        return self._read_state()

    def close(self) -> None:
        self._center_steering()
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None

        self._unhook_memory()

        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def _restart_dolphin(self) -> None:
        self._terminate_existing_process()
        self._process = self._launch_dolphin()
        time.sleep(self.config.post_launch_delay_s)
        self._window_handle = self._wait_for_window()
        self._capture_region = self.config.capture.region or _get_client_rect(
            self._window_handle
        )
        self._init_camera()
        self._hook_memory()
        self._warmup_capture()

    def _terminate_existing_process(self) -> None:
        self._unhook_memory()
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None

        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

        self._process = None
        self._window_handle = None
        self._capture_region = None

    def _launch_dolphin(self) -> subprocess.Popen[str]:
        launch = self.config.launch
        if not launch.dolphin_path.exists():
            raise DolphinDriverError(
                f"Dolphin executable not found: {launch.dolphin_path}"
            )
        if not launch.game_path.exists():
            raise DolphinDriverError(f"Game path not found: {launch.game_path}")
        if launch.save_state_path is not None and not launch.save_state_path.exists():
            raise DolphinDriverError(
                f"Save state path not found: {launch.save_state_path}"
            )

        command = [str(launch.dolphin_path), "--exec", str(launch.game_path)]
        if launch.save_state_path is not None:
            command.extend(["--save_state", str(launch.save_state_path)])
        if launch.batch_mode:
            command.append("--batch")
        if launch.user_path is not None:
            command.extend(["--user", str(launch.user_path)])
        if launch.render_to_main:
            command.extend(["--config", "Dolphin.Display.RenderToMain=True"])

        return subprocess.Popen(command)

    def _ensure_runtime_ready(self) -> None:
        if self._process is None:
            raise DolphinDriverError("Dolphin is not running. Call reset() first.")
        if self._process.poll() is not None:
            raise DolphinDriverError("Dolphin process exited unexpectedly.")
        if self._window_handle is None or self._capture_region is None:
            raise DolphinDriverError("Dolphin window is not ready.")

    def _wait_for_window(self) -> int:
        deadline = time.time() + self.config.launch.launch_timeout_s
        stable_since: float | None = None
        last_handle: int | None = None

        while time.time() < deadline:
            handle = _find_window(self.config.launch.window_title_contains)
            if handle is None:
                stable_since = None
                time.sleep(0.1)
                continue

            if handle != last_handle:
                last_handle = handle
                stable_since = time.time()
            elif stable_since is not None and (
                time.time() - stable_since
            ) >= self.config.launch.stable_window_time_s:
                return handle
            time.sleep(0.1)

        raise DolphinDriverError(
            "Timed out waiting for the Dolphin window to appear."
        )

    def _init_camera(self) -> None:
        try:
            import dxcam  # type: ignore
        except ImportError as exc:
            raise DolphinDriverError(
                "DXcam is not installed. Install with `pip install -e .[windows-dolphin]`."
            ) from exc

        self._dxcam = dxcam
        try:
            self._camera = dxcam.create(output_color=self.config.capture.output_color)
            self._camera.start(
                region=self._capture_region,
                target_fps=self.config.capture.target_fps,
            )
        except Exception as exc:
            raise DolphinDriverError(
                "DXcam failed to start. Confirm the session has an active display."
            ) from exc

    def _warmup_capture(self) -> None:
        if self._camera is None:
            return
        for _ in range(self.config.capture.warmup_frames):
            frame = self._camera.get_latest_frame()
            if frame is not None:
                break
            time.sleep(0.05)

    def _read_state(self) -> StepState:
        frame = self._capture_frame()
        progress = self._read_memory_value(self.config.memory.progress, default=0.0)
        mission_finished = self._read_memory_flag(
            self.config.memory.mission_finished, default=False
        )
        mission_failed = self._read_memory_flag(
            self.config.memory.mission_failed, default=False
        )

        info = {
            "backend": "dolphin",
            "window_handle": self._window_handle,
            "capture_region": self._capture_region,
        }
        return StepState(
            frame=frame,
            progress=float(progress),
            mission_finished=mission_finished,
            mission_failed=mission_failed,
            info=info,
        )

    def _capture_frame(self) -> np.ndarray:
        if self._camera is None:
            raise DolphinDriverError("Capture camera is not initialized.")

        frame = self._camera.get_latest_frame()
        if frame is None:
            raise DolphinDriverError("DXcam did not return a frame.")
        if frame.ndim != 2 and frame.ndim != 3:
            raise DolphinDriverError(f"Unexpected frame shape from DXcam: {frame.shape}")

        return np.ascontiguousarray(frame)

    def _create_gamepad(self):
        try:
            import vgamepad as vg  # type: ignore
        except ImportError as exc:
            raise DolphinDriverError(
                "vgamepad is not installed. Install with `pip install -e .[windows-dolphin]`."
            ) from exc

        try:
            gamepad = vg.VX360Gamepad()
            gamepad.reset()
            gamepad.update()
            return gamepad
        except Exception as exc:
            raise DolphinDriverError(
                "Failed to create a virtual Xbox 360 gamepad. Confirm ViGEmBus is installed."
            ) from exc

    def _apply_steering(self, action: SteeringAction) -> None:
        magnitude = self.config.left_stick_magnitude
        if action == SteeringAction.LEFT:
            x_value = -magnitude
        elif action == SteeringAction.RIGHT:
            x_value = magnitude
        else:
            x_value = 0.0

        self._gamepad.left_joystick_float(
            x_value_float=x_value,
            y_value_float=0.0,
        )
        self._gamepad.update()

    def _center_steering(self) -> None:
        self._gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self._gamepad.update()

    def _load_memory_engine(self):
        try:
            import dolphin_memory_engine as dme  # type: ignore
        except ImportError as exc:
            raise DolphinDriverError(
                "dolphin-memory-engine is not installed. Install with `pip install -e .[windows-dolphin]`."
            ) from exc
        return dme

    def _hook_memory(self) -> None:
        if not any(
            (
                self.config.memory.progress,
                self.config.memory.mission_finished,
                self.config.memory.mission_failed,
            )
        ):
            return

        try:
            self._memory.hook()
        except Exception as exc:
            raise DolphinDriverError(
                "Failed to hook Dolphin memory. Confirm Dolphin is running and visible."
            ) from exc

    def _unhook_memory(self) -> None:
        try:
            if self._memory.is_hooked():
                self._memory.un_hook()
        except Exception:
            pass

    def _read_memory_value(
        self,
        spec: MemoryValueSpec | None,
        *,
        default: float,
    ) -> float:
        if spec is None:
            return default
        self._ensure_memory_ready()
        address = self._resolve_address(spec)
        return float(_read_scalar(self._memory, spec.value_type, address))

    def _read_memory_flag(
        self,
        spec: MemoryFlagSpec | None,
        *,
        default: bool,
    ) -> bool:
        if spec is None:
            return default
        value = self._read_memory_value(spec, default=float(spec.expected_value))
        return value == float(spec.expected_value)

    def _resolve_address(self, spec: MemoryValueSpec) -> int:
        if not spec.pointer_offsets:
            return spec.base_address
        return int(
            self._memory.follow_pointers(spec.base_address, list(spec.pointer_offsets))
        )

    def _ensure_memory_ready(self) -> None:
        if not self._memory.is_hooked():
            raise DolphinDriverError(
                "Dolphin memory is not hooked. Provide memory bindings and call reset()."
            )


def _read_scalar(memory_module, value_type: ScalarType, address: int) -> int | float:
    if value_type == "byte":
        return int(memory_module.read_byte(address))
    if value_type == "word":
        return int(memory_module.read_word(address))
    if value_type == "float":
        return float(memory_module.read_float(address))
    if value_type == "double":
        return float(memory_module.read_double(address))
    raise DolphinDriverError(f"Unsupported memory scalar type: {value_type}")


WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)


def _find_window(title_contains: str) -> int | None:
    user32 = ctypes.windll.user32
    matches: list[int] = []
    title_filter = title_contains.lower()

    @WNDENUMPROC
    def enum_windows(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if title_filter in buffer.value.lower():
            matches.append(int(hwnd))
        return True

    user32.EnumWindows(enum_windows, 0)
    return matches[0] if matches else None


def _get_client_rect(hwnd: int) -> tuple[int, int, int, int]:
    user32 = ctypes.windll.user32
    rect = ctypes.wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise DolphinDriverError("Failed to query the Dolphin client rect.")

    top_left = ctypes.wintypes.POINT(rect.left, rect.top)
    bottom_right = ctypes.wintypes.POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        raise DolphinDriverError("Failed to translate Dolphin client rect.")
    if not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        raise DolphinDriverError("Failed to translate Dolphin client rect.")

    return (
        top_left.x,
        top_left.y,
        bottom_right.x,
        bottom_right.y,
    )
