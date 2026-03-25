from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from sms_rl.config import EnvConfig, EpisodeConfig, ObservationConfig
from sms_rl.courses import BLOOPER_SURFING_WAYPOINTS
from sms_rl.drivers.dolphin import (
    CaptureConfig,
    DolphinDriverConfig,
    DolphinLaunchConfig,
    DolphinWindowsDriver,
    MemoryBindings,
    MemoryFlagSpec,
    MemoryValueSpec,
    sunshine_position_memory_bindings,
)
from sms_rl.envs.blooper_surfing import BlooperSurfingEnv


def parse_address(value: str) -> int:
    return int(value, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump reset and early-step observations to PNG files for visual inspection."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("frame_dump"))
    parser.add_argument("--steps", type=int, default=5)

    parser.add_argument("--obs-width", type=int, default=96)
    parser.add_argument("--obs-height", type=int, default=72)
    parser.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--max-episode-seconds", type=float, default=45.0)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--finish-reward", type=float, default=25.0)
    parser.add_argument("--fail-reward", type=float, default=-25.0)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--path-distance-penalty-scale", type=float, default=0.0)

    parser.add_argument("--dolphin-exe", type=Path, required=True)
    parser.add_argument("--game-path", type=Path, required=True)
    parser.add_argument("--save-state", type=Path)
    parser.add_argument("--user-path", type=Path)
    parser.add_argument(
        "--dolphin-batch-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--window-title", default="Super Mario Sunshine")
    parser.add_argument(
        "--render-to-main",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--control-mode",
        choices=("vgamepad", "keyboard"),
        default="vgamepad",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("dxcam", "mss"),
        default="dxcam",
    )
    parser.add_argument("--capture-fps", type=int, default=30)
    parser.add_argument("--post-launch-delay-seconds", type=float, default=0.0)
    parser.add_argument("--post-reset-delay-seconds", type=float, default=0.0)
    parser.add_argument("--startup-forward-seconds", type=float, default=1.0)
    parser.add_argument("--startup-forward-magnitude", type=float, default=1.0)
    parser.add_argument("--startup-settle-seconds", type=float, default=0.1)
    parser.add_argument("--window-stable-seconds", type=float, default=0.0)

    parser.add_argument("--progress-address", type=parse_address, required=True)
    parser.add_argument(
        "--progress-type",
        choices=("byte", "word", "float", "double"),
        default="float",
    )
    parser.add_argument("--finished-address", type=parse_address, required=True)
    parser.add_argument(
        "--finished-type",
        choices=("byte", "word", "float", "double"),
        default="byte",
    )
    parser.add_argument("--finished-value", type=float, default=1.0)
    parser.add_argument("--failed-address", type=parse_address, required=True)
    parser.add_argument(
        "--failed-type",
        choices=("byte", "word", "float", "double"),
        default="byte",
    )
    parser.add_argument("--failed-value", type=float, default=1.0)
    return parser


def build_env(args: argparse.Namespace) -> BlooperSurfingEnv:
    env_config = EnvConfig(
        observation=ObservationConfig(
            width=args.obs_width,
            height=args.obs_height,
            grayscale=args.grayscale,
            frame_stack=args.frame_stack,
        ),
        episode=EpisodeConfig(
            max_steps=args.max_steps,
            max_episode_seconds=args.max_episode_seconds,
            action_repeat=args.action_repeat,
            step_penalty=args.step_penalty,
            finish_reward=args.finish_reward,
            fail_reward=args.fail_reward,
            progress_reward_scale=args.progress_reward_scale,
            path_distance_penalty_scale=args.path_distance_penalty_scale,
            path_waypoints=BLOOPER_SURFING_WAYPOINTS,
        ),
    )
    memory = MemoryBindings(
        progress=MemoryValueSpec(
            base_address=args.progress_address,
            value_type=args.progress_type,
        ),
        mission_finished=MemoryFlagSpec(
            base_address=args.finished_address,
            value_type=args.finished_type,
            expected_value=args.finished_value,
        ),
        mission_failed=MemoryFlagSpec(
            base_address=args.failed_address,
            value_type=args.failed_type,
            expected_value=args.failed_value,
        ),
        **sunshine_position_memory_bindings(),
    )
    driver = DolphinWindowsDriver(
        DolphinDriverConfig(
            launch=DolphinLaunchConfig(
                dolphin_path=args.dolphin_exe,
                game_path=args.game_path,
                save_state_path=args.save_state,
                batch_mode=args.dolphin_batch_mode,
                user_path=args.user_path,
                window_title_contains=args.window_title,
                stable_window_time_s=args.window_stable_seconds,
                render_to_main=args.render_to_main,
            ),
            capture=CaptureConfig(
                target_fps=args.capture_fps,
                backend=args.capture_backend,
            ),
            memory=memory,
            control_mode=args.control_mode,
            post_launch_delay_s=args.post_launch_delay_seconds,
            post_reset_delay_s=args.post_reset_delay_seconds,
            startup_forward_seconds=args.startup_forward_seconds,
            startup_forward_magnitude=args.startup_forward_magnitude,
            startup_settle_seconds=args.startup_settle_seconds,
        )
    )
    return BlooperSurfingEnv(config=env_config, driver=driver)


def obs_to_frame(observation: np.ndarray, grayscale: bool) -> np.ndarray:
    if grayscale:
        last = observation[-1]
        return np.repeat(last[:, :, None], 3, axis=2)
    channels_per_frame = 3
    frame = observation[-channels_per_frame:, :, :]
    return np.transpose(frame, (1, 2, 0))


def save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)


def main() -> None:
    args = build_parser().parse_args()
    env = build_env(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        obs, info = env.reset()
        save_frame(output_dir / "reset_obs.png", obs_to_frame(obs, args.grayscale))
        (output_dir / "reset_info.txt").write_text(f"{info}\n", encoding="utf-8")

        terminated = False
        truncated = False
        action_cycle = [0, 1, 2, 1]
        for step_idx in range(args.steps):
            if terminated or truncated:
                break
            action = action_cycle[step_idx % len(action_cycle)]
            obs, reward, terminated, truncated, info = env.step(action)
            save_frame(
                output_dir / f"step_{step_idx:02d}_action_{action}.png",
                obs_to_frame(obs, args.grayscale),
            )
            (output_dir / f"step_{step_idx:02d}_info.txt").write_text(
                f"reward={reward}\nterminated={terminated}\ntruncated={truncated}\ninfo={info}\n",
                encoding="utf-8",
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
