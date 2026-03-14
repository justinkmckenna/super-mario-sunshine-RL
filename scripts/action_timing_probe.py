from __future__ import annotations

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

from sms_rl.config import EnvConfig, EpisodeConfig, ObservationConfig
from sms_rl.drivers.dolphin import (
    CaptureConfig,
    DolphinDriverConfig,
    DolphinLaunchConfig,
    DolphinWindowsDriver,
    MemoryBindings,
    MemoryFlagSpec,
    MemoryValueSpec,
)
from sms_rl.envs.blooper_surfing import BlooperSurfingEnv


def parse_address(value: str) -> int:
    return int(value, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe action-decision timing and log per-step latency to CSV."
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=60)
    parser.add_argument("--probe-seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("action_timing_probe.csv"))
    parser.add_argument(
        "--log-every-step",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
        "--control-mode",
        choices=("vgamepad", "keyboard"),
        default="vgamepad",
    )
    parser.add_argument("--capture-fps", type=int, default=30)
    parser.add_argument("--post-launch-delay-seconds", type=float, default=0.5)
    parser.add_argument("--post-reset-delay-seconds", type=float, default=0.05)
    parser.add_argument("--window-stable-seconds", type=float, default=0.2)
    parser.add_argument(
        "--pause-on-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause emulator at reset completion; first step unpauses then applies action.",
    )
    parser.add_argument(
        "--restart-on-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-state-slot", type=int, default=1)
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
            ),
            capture=CaptureConfig(target_fps=args.capture_fps),
            memory=memory,
            control_mode=args.control_mode,
            restart_on_reset=args.restart_on_reset,
            save_state_slot=args.save_state_slot,
            post_launch_delay_s=args.post_launch_delay_seconds,
            post_reset_delay_s=args.post_reset_delay_seconds,
            pause_on_reset=args.pause_on_reset,
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
    run_start = time.perf_counter()
    cumulative_step_wall_s = 0.0
    reset_overheads: list[float] = []

    try:
        for episode_idx in range(args.episodes):
            reset_start = time.perf_counter()
            obs, info = env.reset()
            reset_end = time.perf_counter()
            reset_progress = float(info.get("progress", 0.0))
            reset_failed = bool(info.get("mission_failed", False))
            reset_finished = bool(info.get("mission_finished", False))
            env_reset_elapsed = float(info.get("env_reset_elapsed_s", 0.0))
            driver_reset_elapsed = float(info.get("driver_reset_elapsed_s", 0.0))
            used_soft_reset = bool(info.get("driver_reset_used_soft_reset", False))
            recovered_relaunch = bool(
                info.get("driver_reset_recovered_via_relaunch", False)
            )
            print(
                f"[episode {episode_idx}] reset_wall_s={reset_end - reset_start:.3f} "
                f"env_reset_s={env_reset_elapsed:.3f} "
                f"driver_reset_s={driver_reset_elapsed:.3f} "
                f"soft_reset={used_soft_reset} relaunch_recovery={recovered_relaunch} "
                f"progress={reset_progress:.6f} "
                f"finished={reset_finished} failed={reset_failed}",
                flush=True,
            )
            del obs, info
            episode_start = time.perf_counter()
            reset_overheads.append(reset_end - reset_start)
            terminated = False
            truncated = False
            last_action: int | None = None
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

                if args.log_every_step or action != last_action:
                    print(
                        f"[episode {episode_idx}] step={decision_in_episode} t={elapsed_ep_s:.3f}s action={action}",
                        flush=True,
                    )
                    last_action = action

                _, reward, terminated, truncated, info = env.step(action)
                t1 = time.perf_counter()
                step_dt = t1 - t0
                total_decisions += 1
                cumulative_step_wall_s += step_dt
                rows.append(
                    [
                        datetime.now().isoformat(),
                        episode_idx,
                        total_decisions,
                        action,
                        round(t0 - run_start, 6),
                        round(step_dt, 6),
                        round(t1 - episode_start, 6),
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
                "wall_time_iso",
                "episode",
                "decision_idx",
                "action",
                "since_run_start_s",
                "step_wall_dt_s",
                "since_episode_start_s",
                "reward",
                "terminated",
                "truncated",
                "mission_finished",
                "mission_failed",
                "progress",
            ]
        )
        writer.writerows(rows)

    elapsed = time.perf_counter() - run_start
    dps_end_to_end = (total_decisions / elapsed) if elapsed > 0 else 0.0
    dps_active = (
        (total_decisions / cumulative_step_wall_s)
        if cumulative_step_wall_s > 0
        else 0.0
    )
    reset_total = sum(reset_overheads)
    reset_mean = (reset_total / len(reset_overheads)) if reset_overheads else 0.0
    print(f"Wrote: {args.output_csv}")
    print(f"Decisions: {total_decisions}")
    print(f"Elapsed_s: {elapsed:.3f}")
    print(f"Decisions_per_second_end_to_end: {dps_end_to_end:.3f}")
    print(f"Active_step_time_s: {cumulative_step_wall_s:.3f}")
    print(f"Decisions_per_second_active: {dps_active:.3f}")
    print(f"Reset_count: {len(reset_overheads)}")
    print(f"Reset_total_s: {reset_total:.3f}")
    print(f"Reset_mean_s: {reset_mean:.3f}")


if __name__ == "__main__":
    main()
