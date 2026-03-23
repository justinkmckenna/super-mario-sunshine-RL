from __future__ import annotations

import argparse
from pathlib import Path

from sms_rl.baselines import CenteringPolicy, ConstantPolicy, RandomPolicy, run_episode
from sms_rl.config import EnvConfig, EpisodeConfig
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
    parser = argparse.ArgumentParser(description="Run Blooper Surfing smoke tests.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--backend", choices=("mock", "dolphin"), default="mock")
    parser.add_argument(
        "--baseline",
        choices=("random", "scripted", "neutral"),
        default="scripted",
    )
    parser.add_argument("--dolphin-exe", type=Path)
    parser.add_argument("--game-path", type=Path)
    parser.add_argument("--save-state", type=Path)
    parser.add_argument("--user-path", type=Path)
    parser.add_argument(
        "--dolphin-batch-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch Dolphin with --batch to suppress interactive emulator prompts.",
    )
    parser.add_argument("--window-title", default="Dolphin")
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
    parser.add_argument("--capture-fps", type=int, default=60)
    parser.add_argument(
        "--restart-on-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, reset episodes by loading Dolphin savestate slot instead of relaunching Dolphin.",
    )
    parser.add_argument("--save-state-slot", type=int, default=1)
    parser.add_argument("--progress-address", type=parse_address)
    parser.add_argument(
        "--progress-type",
        choices=("byte", "word", "float", "double"),
        default="float",
    )
    parser.add_argument("--finished-address", type=parse_address)
    parser.add_argument(
        "--finished-type",
        choices=("byte", "word", "float", "double"),
        default="byte",
    )
    parser.add_argument("--finished-value", type=float, default=1.0)
    parser.add_argument("--failed-address", type=parse_address)
    parser.add_argument(
        "--failed-type",
        choices=("byte", "word", "float", "double"),
        default="byte",
    )
    parser.add_argument("--failed-value", type=float, default=1.0)
    return parser


def build_env(args: argparse.Namespace) -> BlooperSurfingEnv:
    if args.backend == "mock":
        return BlooperSurfingEnv(
            config=EnvConfig(
                episode=EpisodeConfig(
                    path_waypoints=BLOOPER_SURFING_WAYPOINTS,
                )
            )
        )

    if args.dolphin_exe is None or args.game_path is None:
        raise SystemExit("--dolphin-exe and --game-path are required for --backend dolphin.")

    memory = MemoryBindings(
        progress=(
            MemoryValueSpec(
                base_address=args.progress_address,
                value_type=args.progress_type,
            )
            if args.progress_address is not None
            else None
        ),
        mission_finished=(
            MemoryFlagSpec(
                base_address=args.finished_address,
                value_type=args.finished_type,
                expected_value=args.finished_value,
            )
            if args.finished_address is not None
            else None
        ),
        mission_failed=(
            MemoryFlagSpec(
                base_address=args.failed_address,
                value_type=args.failed_type,
                expected_value=args.failed_value,
            )
            if args.failed_address is not None
            else None
        ),
        **sunshine_position_memory_bindings(),
    )
    driver_config = DolphinDriverConfig(
        launch=DolphinLaunchConfig(
            dolphin_path=args.dolphin_exe,
            game_path=args.game_path,
            save_state_path=args.save_state,
            batch_mode=args.dolphin_batch_mode,
            user_path=args.user_path,
            window_title_contains=args.window_title,
            render_to_main=args.render_to_main,
        ),
        capture=CaptureConfig(
            target_fps=args.capture_fps,
            backend=args.capture_backend,
        ),
        memory=memory,
        control_mode=args.control_mode,
    )
    driver = DolphinWindowsDriver(driver_config)
    return BlooperSurfingEnv(
        config=EnvConfig(
            episode=EpisodeConfig(
                path_waypoints=BLOOPER_SURFING_WAYPOINTS,
            )
        ),
        driver=driver,
    )


def build_policy(args: argparse.Namespace, env: BlooperSurfingEnv):
    if args.baseline == "random":
        return RandomPolicy(env.action_space.n, seed=0)
    if args.baseline == "neutral":
        return ConstantPolicy(action=1)
    if args.backend == "dolphin":
        raise SystemExit("The scripted baseline only supports the mock backend right now.")
    return CenteringPolicy()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    env = build_env(args)
    try:
        policy = build_policy(args, env)

        for episode_idx in range(args.episodes):
            result = run_episode(env, policy)
            print(
                "episode="
                f"{episode_idx} reward={result.total_reward:.2f} steps={result.steps} "
                f"terminated={result.terminated} truncated={result.truncated} "
                f"finished={result.info.get('mission_finished', False)} "
                f"failed={result.info.get('mission_failed', False)}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
