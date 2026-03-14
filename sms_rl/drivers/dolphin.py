from __future__ import annotations

import contextlib
import ctypes
import gc
import io
import os
import stat
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    launch_timeout_s: float = 30.0
    stable_window_time_s: float = 1.0


@dataclass(slots=True)
class CaptureConfig:
    region: tuple[int, int, int, int] | None = None
    output_color: Literal["gray", "rgb", "bgr"] = "gray"
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
    control_mode: Literal["vgamepad", "keyboard"] = "vgamepad"
    restart_on_reset: bool = True
    neutral_deadzone: float = 0.2
    left_stick_magnitude: float = 0.75
    post_launch_delay_s: float = 2.0
    post_reset_delay_s: float = 0.5
    step_sleep_s: float = 0.0
    step_frame_time_s: float = 0.05
    keyboard_left_vk: int = 0x25   # Left arrow
    keyboard_right_vk: int = 0x27  # Right arrow
    keyboard_jump_vk: int = 0x58   # X key
    save_state_slot: int = 1
    initialize_reset_slot_on_launch: bool = True
    post_soft_reset_delay_s: float = 0.35
    soft_reset_attempts: int = 3
    soft_reset_progress_tolerance: float = 5.0
    launch_retries: int = 4
    launch_retry_backoff_s: float = 0.75
    pause_on_reset: bool = False
    pause_toggle_vk: int = 0x79  # F10 by default in Dolphin
    post_pause_toggle_delay_s: float = 0.05


class DolphinWindowsDriver:
    """Windows Dolphin driver backed by DXcam and Dolphin Memory Engine.

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
        self._last_frame: np.ndarray | None = None
        self._gamepad = self._create_gamepad() if self._use_vgamepad else None
        self._memory = self._load_memory_engine()
        self._dxcam = None
        self._slot_initialized = False
        self._expected_reset_progress: float | None = None
        self._needs_unpause_on_first_step = False

    @property
    def _use_vgamepad(self) -> bool:
        return self.config.control_mode == "vgamepad"

    def reset(self) -> StepState:
        used_soft_reset = False
        if self.config.restart_on_reset or self._process is None:
            self._restart_dolphin()
        else:
            try:
                self._soft_reset_to_start()
                used_soft_reset = True
            except Exception:
                # Fall back to full relaunch if in-process reset fails.
                self._restart_dolphin()

        self._focus_window()
        self._center_steering()
        time.sleep(self.config.post_reset_delay_s)
        try:
            return self._read_state()
        except Exception:
            # If soft reset left capture/memory in a bad state, recover by relaunching.
            if used_soft_reset:
                self._restart_dolphin()
                self._focus_window()
                self._center_steering()
                time.sleep(self.config.post_reset_delay_s)
                return self._read_state()
            raise

    def step(self, action: SteeringAction, repeat: int) -> StepState:
        self._ensure_runtime_ready()
        self._focus_window()
        if self._needs_unpause_on_first_step:
            self._toggle_pause()
            time.sleep(self.config.post_pause_toggle_delay_s)
            self._needs_unpause_on_first_step = False
        repeat_count = max(1, repeat)
        for _ in range(repeat_count):
            self._apply_steering(action)
            sleep_s = (
                self.config.step_sleep_s
                if self.config.step_sleep_s > 0
                else self.config.step_frame_time_s
            )
            if sleep_s > 0:
                time.sleep(sleep_s)
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
            self._shutdown_process_gracefully()
        self._process = None

    def _restart_dolphin(self) -> None:
        self._terminate_existing_process()
        retries = max(1, self.config.launch_retries)
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                self._process = self._launch_dolphin()
                time.sleep(self.config.post_launch_delay_s + (attempt * 0.15))
                self._window_handle = self._wait_for_window()
                self._capture_region = self.config.capture.region or _get_client_rect(
                    self._window_handle
                )
                if self.config.pause_on_reset:
                    self._focus_window()
                    self._toggle_pause()
                    time.sleep(self.config.post_pause_toggle_delay_s)
                    self._needs_unpause_on_first_step = True
                self._init_camera()
                self._hook_memory()
                self._warmup_capture()
                self._slot_initialized = False
                if (
                    self.config.launch.save_state_path is not None
                    and self.config.initialize_reset_slot_on_launch
                ):
                    self._initialize_reset_slot()
                try:
                    start_state = self._read_state()
                    self._expected_reset_progress = float(start_state.progress)
                except Exception:
                    self._expected_reset_progress = None
                return
            except Exception as exc:
                last_exc = exc
                self._terminate_existing_process()
                if self.config.launch.user_path is not None:
                    _cleanup_stale_wii_fst_temp(self.config.launch.user_path)
                if attempt < retries - 1:
                    delay = self.config.launch_retry_backoff_s * (attempt + 1)
                    time.sleep(delay)
                    continue
                break
        raise DolphinDriverError(
            f"Failed to relaunch Dolphin after {retries} attempts."
        ) from last_exc

    def _soft_reset_to_start(self) -> None:
        self._ensure_runtime_ready()
        self._focus_window()
        if not self._slot_initialized:
            if self.config.launch.save_state_path is None:
                raise DolphinDriverError(
                    "Cannot soft reset without initial save state. Configure --save-state."
                )
            self._initialize_reset_slot()
        attempts = max(1, self.config.soft_reset_attempts)
        last_exc: Exception | None = None
        for _attempt in range(attempts):
            try:
                self._load_state_slot()
                time.sleep(self.config.post_soft_reset_delay_s)
                # DXGI duplication can become stale on in-process load-state transitions.
                self._recover_camera(hard=True)
                if not self._memory.is_hooked():
                    self._hook_memory()
                state = self._read_state()
                if self._is_soft_reset_state_valid(state):
                    if self.config.pause_on_reset:
                        self._toggle_pause()
                        time.sleep(self.config.post_pause_toggle_delay_s)
                        self._needs_unpause_on_first_step = True
                    return
            except Exception as exc:
                last_exc = exc
            time.sleep(0.08)
        raise DolphinDriverError(
            f"Soft reset to savestate slot failed after {attempts} attempts."
        ) from last_exc

    def _is_soft_reset_state_valid(self, state: StepState) -> bool:
        if state.mission_finished or state.mission_failed:
            return False
        if self._expected_reset_progress is None:
            return True
        return (
            abs(float(state.progress) - self._expected_reset_progress)
            <= self.config.soft_reset_progress_tolerance
        )

    def _initialize_reset_slot(self) -> None:
        self._save_state_slot()
        self._slot_initialized = True

    def _state_slot_vk(self) -> int:
        slot = self.config.save_state_slot
        if slot < 1 or slot > 8:
            raise DolphinDriverError("save_state_slot must be in range 1..8")
        return 0x70 + (slot - 1)  # F1..F8

    def _save_state_slot(self) -> None:
        slot_vk = self._state_slot_vk()
        shift_vk = 0x10
        _key_down(shift_vk)
        try:
            _tap_key(slot_vk, hold_s=0.05)
        finally:
            _key_up(shift_vk)
        time.sleep(0.1)

    def _load_state_slot(self) -> None:
        slot_vk = self._state_slot_vk()
        _tap_key(slot_vk, hold_s=0.05)

    def _terminate_existing_process(self) -> None:
        self._unhook_memory()
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None

        if self._process is not None and self._process.poll() is None:
            self._shutdown_process_gracefully()

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
        if launch.user_path is not None:
            _cleanup_stale_wii_fst_temp(launch.user_path)

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

    def _shutdown_process_gracefully(self) -> None:
        # In non-batch mode we can ask Dolphin to close cleanly through WM_CLOSE.
        # In batch mode, skip WM_CLOSE to avoid any confirmation dialog path.
        try:
            if (not self.config.launch.batch_mode) and self._window_handle is not None:
                _post_close_window(self._window_handle)
        except Exception:
            pass

        if self._process is None:
            return

        try:
            self._process.wait(timeout=4)
            return
        except subprocess.TimeoutExpired:
            pass

        self._process.terminate()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()

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
        self._camera = self._create_camera()

    def _warmup_capture(self) -> None:
        if self._camera is None:
            return
        for _ in range(self.config.capture.warmup_frames):
            frame = self._camera.grab(region=self._capture_region)
            if frame is not None:
                break
            time.sleep(0.05)

    def _read_state(self) -> StepState:
        frame = self._capture_frame()
        info = {
            "backend": "dolphin",
            "window_handle": self._window_handle,
            "capture_region": self._capture_region,
        }
        try:
            progress = self._read_memory_value(self.config.memory.progress, default=0.0)
            mission_finished = self._read_memory_flag(
                self.config.memory.mission_finished, default=False
            )
            mission_failed = self._read_memory_flag(
                self.config.memory.mission_failed, default=False
            )
        except Exception as exc:
            # Fail closed on unrecoverable memory read errors so long runs do not hang/crash.
            info["memory_error"] = str(exc)
            progress = 0.0
            mission_finished = False
            mission_failed = True
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

        frame = None
        consecutive_grab_errors = 0
        for _attempt in range(12):
            try:
                frame = self._camera.grab(region=self._capture_region)
            except Exception:
                # DXGI duplication can transiently fail (e.g. keyed mutex abandoned).
                # Only perform hard recovery after repeated failures.
                consecutive_grab_errors += 1
                if consecutive_grab_errors >= 3:
                    self._recover_camera(hard=True)
                    consecutive_grab_errors = 0
                time.sleep(0.03)
                continue
            if frame is not None:
                break
            # `None` frames can occur transiently; do not recreate the camera.
            time.sleep(0.01)
        if frame is None:
            if self._last_frame is not None:
                return self._last_frame.copy()
            raise DolphinDriverError("DXcam did not return a frame.")
        if frame.ndim != 2 and frame.ndim != 3:
            raise DolphinDriverError(f"Unexpected frame shape from DXcam: {frame.shape}")

        contiguous = np.ascontiguousarray(frame)
        self._last_frame = contiguous
        return contiguous

    def _recover_camera(self, *, hard: bool = False) -> None:
        if self._dxcam is None:
            raise DolphinDriverError("DXcam module is not initialized.")
        if self._capture_region is None:
            raise DolphinDriverError("Capture region is not initialized.")

        if self._camera is None:
            self._camera = self._create_camera()
            return

        if not hard:
            on_output_change = getattr(self._camera, "_on_output_change", None)
            if callable(on_output_change):
                try:
                    on_output_change()
                    return
                except Exception:
                    pass

        try:
            self._camera.release()
        except Exception:
            pass
        self._camera = None
        gc.collect()
        self._camera = self._create_camera()

    def _create_camera(self):
        if self._dxcam is None:
            raise DolphinDriverError("DXcam module is not initialized.")
        output_color = _to_dxcam_color(self.config.capture.output_color)
        muted_out = io.StringIO()
        try:
            # dxcam prints singleton messages to stdout; suppress to keep logs clean.
            with contextlib.redirect_stdout(muted_out), contextlib.redirect_stderr(
                muted_out
            ):
                return self._dxcam.create(output_color=output_color)
        except Exception as exc:
            raise DolphinDriverError(
                "DXcam failed to start. Confirm the session has an active display."
            ) from exc

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
        if not self._use_vgamepad:
            self._apply_keyboard_steering(action)
            return

        if self._gamepad is None:
            raise DolphinDriverError("Virtual gamepad was not initialized.")
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
        # Optional vgamepad jump behavior if this mode is used later.
        if action == SteeringAction.JUMP:
            import vgamepad as vg  # type: ignore

            self._gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        else:
            import vgamepad as vg  # type: ignore

            self._gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self._gamepad.update()

    def _center_steering(self) -> None:
        if self._use_vgamepad:
            if self._gamepad is None:
                return
            self._gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self._gamepad.update()
            return
        _key_up(self.config.keyboard_left_vk)
        _key_up(self.config.keyboard_right_vk)
        _key_up(self.config.keyboard_jump_vk)

    def _apply_keyboard_steering(self, action: SteeringAction) -> None:
        if action == SteeringAction.LEFT:
            _key_down(self.config.keyboard_left_vk)
            _key_up(self.config.keyboard_right_vk)
            _key_up(self.config.keyboard_jump_vk)
        elif action == SteeringAction.RIGHT:
            _key_down(self.config.keyboard_right_vk)
            _key_up(self.config.keyboard_left_vk)
            _key_up(self.config.keyboard_jump_vk)
        elif action == SteeringAction.JUMP:
            _key_up(self.config.keyboard_left_vk)
            _key_up(self.config.keyboard_right_vk)
            _tap_key(self.config.keyboard_jump_vk)
        else:
            _key_up(self.config.keyboard_left_vk)
            _key_up(self.config.keyboard_right_vk)
            _key_up(self.config.keyboard_jump_vk)

    def _focus_window(self) -> None:
        if self._window_handle is None:
            return
        user32 = ctypes.windll.user32
        user32.SetForegroundWindow(self._window_handle)

    def _toggle_pause(self) -> None:
        _tap_key(self.config.pause_toggle_vk, hold_s=0.04)

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
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                self._ensure_memory_ready()
                address = self._resolve_address(spec)
                return float(_read_scalar(self._memory, spec.value_type, address))
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    self._rehook_memory()
                    time.sleep(0.02)
                    continue
                break
        raise DolphinDriverError(
            f"Failed to read Dolphin memory at {hex(spec.base_address)}"
        ) from last_exc

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

    def _rehook_memory(self) -> None:
        try:
            if self._memory.is_hooked():
                self._memory.un_hook()
        except Exception:
            pass
        self._memory.hook()


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


def _to_dxcam_color(value: str) -> str:
    # DXcam color modes are case-sensitive and expected in uppercase.
    mapping = {
        "gray": "GRAY",
        "rgb": "RGB",
        "bgr": "BGR",
    }
    key = value.lower()
    if key not in mapping:
        raise DolphinDriverError(f"Unsupported capture output_color: {value}")
    return mapping[key]


def _key_down(vk_code: int) -> None:
    _send_input_key(vk_code, key_up=False)


def _key_up(vk_code: int) -> None:
    _send_input_key(vk_code, key_up=True)


def _tap_key(vk_code: int, hold_s: float = 0.03) -> None:
    _key_down(vk_code)
    time.sleep(hold_s)
    _key_up(vk_code)


def _send_input_key(vk_code: int, *, key_up: bool) -> None:
    user32 = ctypes.windll.user32

    INPUT_KEYBOARD = 1
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008

    scan = user32.MapVirtualKeyW(vk_code, 0)
    if scan == 0:
        return

    flags = KEYEVENTF_SCANCODE
    if vk_code in (0x25, 0x26, 0x27, 0x28):  # Arrow keys.
        flags |= KEYEVENTF_EXTENDEDKEY
    if key_up:
        flags |= KEYEVENTF_KEYUP

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", wintypes.DWORD),
            ("ki", KEYBDINPUT),
        ]

    payload = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(
            wVk=0,
            wScan=scan,
            dwFlags=flags,
            time=0,
            dwExtraInfo=None,
        ),
    )
    user32.SendInput(1, ctypes.byref(payload), ctypes.sizeof(INPUT))


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
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise DolphinDriverError("Failed to query the Dolphin client rect.")

    top_left = wintypes.POINT(rect.left, rect.top)
    bottom_right = wintypes.POINT(rect.right, rect.bottom)
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


def _post_close_window(hwnd: int) -> None:
    user32 = ctypes.windll.user32
    WM_CLOSE = 0x0010
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)


def _cleanup_stale_wii_fst_temp(user_path: Path) -> None:
    # Dolphin can leave stale temp files like `fst.bin.xxx` on abrupt shutdown.
    # If present, next launch may show a modal warning and block automation.
    wii_dir = user_path / "Wii"
    if not wii_dir.exists():
        return
    candidates = [wii_dir / "fst.bin"]
    candidates.extend(list(wii_dir.glob("fst.bin.*")))
    for candidate in candidates:
        # Retry because Dolphin may release file handles shortly after exit.
        for _ in range(10):
            try:
                if candidate.exists():
                    os.chmod(candidate, stat.S_IWRITE)
                    candidate.unlink()
                break
            except OSError:
                time.sleep(0.15)
