from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

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
        description="Probe action decisions and log per-step outcomes to CSV."
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=60)
    parser.add_argument("--probe-seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("action_timing_probe.csv"))
    parser.add_argument(
        "--policy",
        choices=("random", "scripted-midtest"),
        default="random",
        help="random: random actions. scripted-midtest: deterministic half-second pulse pattern.",
    )

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
    parser.add_argument("--post-launch-delay-seconds", type=float, default=0.5)
    parser.add_argument("--post-reset-delay-seconds", type=float, default=0.05)
    parser.add_argument("--startup-forward-seconds", type=float, default=0.0)
    parser.add_argument("--startup-forward-magnitude", type=float, default=1.0)
    parser.add_argument("--startup-settle-seconds", type=float, default=0.1)
    parser.add_argument("--window-stable-seconds", type=float, default=0.2)
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


def scripted_midtest_action(decision_in_episode: int, elapsed_s: float) -> int:
    # Action map: 0=LEFT, 1=NEUTRAL, 2=RIGHT, 3=JUMP
    # Requested probe pattern:
    # 0.5s LEFT for 0.2s, 1.0s RIGHT for 0.2s, ... alternating until 4.0s.
    # Outside pulse windows: NEUTRAL.
    del decision_in_episode
    first_pulse_s = 0.5
    pulse_interval_s = 0.5
    pulse_width_s = 0.2
    pulse_count = 8
    for pulse_idx in range(pulse_count):
        start_s = first_pulse_s + pulse_idx * pulse_interval_s
        end_s = start_s + pulse_width_s
        if start_s <= elapsed_s < end_s:
            return 0 if (pulse_idx % 2 == 0) else 2
    return 1


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    env = build_env(args)
    rows: list[list[object]] = []
    total_decisions = 0

    try:
        for episode_idx in range(args.episodes):
            obs, info = env.reset()
            del obs, info
            episode_start = time.perf_counter()
            terminated = False
            truncated = False
            decision_in_episode = 0

            while not terminated and not truncated and total_decisions < args.max_decisions:
                t0 = time.perf_counter()
                elapsed_ep_s = t0 - episode_start
                if elapsed_ep_s >= args.probe_seconds:
                    break

                decision_in_episode += 1
                if args.policy == "scripted-midtest":
                    action = scripted_midtest_action(decision_in_episode, elapsed_ep_s)
                else:
                    action = rng.randrange(env.action_space.n)

                _, reward, terminated, truncated, info = env.step(action)
                total_decisions += 1
                rows.append(
                    [
                        episode_idx,
                        total_decisions,
                        action,
                        round(float(reward), 6),
                        bool(terminated),
                        bool(truncated),
                        bool(info.get("mission_finished", False)),
                        bool(info.get("mission_failed", False)),
                        round(float(info.get("progress", 0.0)), 6),
                    ]
                )

            if total_decisions >= args.max_decisions:
                break
    finally:
        env.close()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode",
                "decision_idx",
                "action",
                "reward",
                "terminated",
                "truncated",
                "mission_finished",
                "mission_failed",
                "progress",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
