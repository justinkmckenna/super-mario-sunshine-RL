from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
    parser = argparse.ArgumentParser(description="Train PPO on Blooper Surfing.")
    parser.add_argument("--run-name", default=f"ppo_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument(
        "--record-eval-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-video-fps", type=int, default=30)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-checkpoint", type=Path)

    parser.add_argument("--obs-width", type=int, default=96)
    parser.add_argument("--obs-height", type=int, default=72)
    parser.add_argument(
        "--grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--max-episode-seconds", type=float, default=45.0)
    parser.add_argument("--action-repeat", type=int, default=4)
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


def build_env_factory(args: argparse.Namespace) -> Callable[[], BlooperSurfingEnv]:
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

    def make_env() -> BlooperSurfingEnv:
        driver_config = DolphinDriverConfig(
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
        return BlooperSurfingEnv(config=env_config, driver=DolphinWindowsDriver(driver_config))

    return make_env


def obs_to_frame(observation: np.ndarray, grayscale: bool) -> np.ndarray:
    # Convert stacked observation to a single RGB frame for video export.
    if grayscale:
        last = observation[-1]
        return np.repeat(last[:, :, None], 3, axis=2)
    channels_per_frame = 3
    frame = observation[-channels_per_frame:, :, :]
    return np.transpose(frame, (1, 2, 0))


def evaluate_model(
    model: Any,
    env: Any,
    eval_episodes: int,
    *,
    deterministic: bool = True,
    record_video: bool,
    video_path: Path | None,
    video_fps: int,
    grayscale: bool,
) -> dict[str, float]:
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    success_count = 0
    fail_count = 0
    captured_frames: list[np.ndarray] = []
    for episode_idx in range(eval_episodes):
        obs, info = env.reset()
        del info
        terminated = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0
        if record_video and episode_idx == 0:
            captured_frames.append(obs_to_frame(obs, grayscale))

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            ep_steps += 1
            if record_video and episode_idx == 0:
                captured_frames.append(obs_to_frame(obs, grayscale))

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        success_count += int(bool(info.get("mission_finished", False)))
        fail_count += int(bool(info.get("mission_failed", False)))

    if record_video and video_path is not None and captured_frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        with imageio.get_writer(video_path, fps=video_fps) as writer:
            for frame in captured_frames:
                writer.append_data(frame)

    episodes = max(1, len(episode_rewards))
    return {
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "success_rate": success_count / episodes,
        "fail_rate": fail_count / episodes,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    run_dir = args.output_dir / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    tensorboard_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    make_env = build_env_factory(args)
    train_env = DummyVecEnv([lambda: Monitor(make_env())])

    if args.resume_checkpoint is not None:
        model = PPO.load(
            str(args.resume_checkpoint),
            env=train_env,
            device=args.device,
        )
        model.learning_rate = args.learning_rate
        if hasattr(model, "lr_schedule"):
            model.lr_schedule = lambda _progress_remaining: args.learning_rate
        model.tensorboard_log = str(tensorboard_dir)
        model.verbose = 1
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.n_epochs = args.n_epochs
        model.gamma = args.gamma
        model.gae_lambda = args.gae_lambda
        model.ent_coef = args.ent_coef
        model.vf_coef = args.vf_coef
    else:
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            device=args.device,
            seed=args.seed,
        )

    eval_csv_path = eval_dir / "eval_metrics.csv"
    with eval_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timesteps",
                "mean_reward",
                "mean_length",
                "success_rate",
                "fail_rate",
                "checkpoint_path",
                "video_path",
            ]
        )

    next_eval = max(1, args.eval_every)
    next_checkpoint = max(1, args.checkpoint_every)

    try:
        while model.num_timesteps < args.total_timesteps:
            next_target = min(next_eval, next_checkpoint, args.total_timesteps)
            chunk = max(1, next_target - model.num_timesteps)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                tb_log_name=args.run_name,
                progress_bar=False,
            )

            checkpoint_path = ""
            while model.num_timesteps >= next_checkpoint:
                checkpoint_file = checkpoint_dir / f"ppo_step_{next_checkpoint}.zip"
                model.save(checkpoint_file)
                checkpoint_path = str(checkpoint_file)
                print(f"[checkpoint] {checkpoint_file}")
                next_checkpoint += max(1, args.checkpoint_every)

            while model.num_timesteps >= next_eval:
                eval_env = train_env.envs[0]
                video_path = (
                    eval_dir / f"eval_step_{next_eval}.mp4"
                    if args.record_eval_video
                    else None
                )
                metrics = evaluate_model(
                    model,
                    eval_env,
                    args.eval_episodes,
                    deterministic=True,
                    record_video=args.record_eval_video,
                    video_path=video_path,
                    video_fps=args.eval_video_fps,
                    grayscale=args.grayscale,
                )
                print(
                    "[eval] "
                    f"timesteps={model.num_timesteps} "
                    f"mean_reward={metrics['mean_reward']:.3f} "
                    f"mean_length={metrics['mean_length']:.2f} "
                    f"success_rate={metrics['success_rate']:.2%} "
                    f"fail_rate={metrics['fail_rate']:.2%}"
                )

                with eval_csv_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            model.num_timesteps,
                            metrics["mean_reward"],
                            metrics["mean_length"],
                            metrics["success_rate"],
                            metrics["fail_rate"],
                            checkpoint_path,
                            str(video_path) if video_path is not None else "",
                        ]
                    )
                train_env.reset()
                next_eval += max(1, args.eval_every)
    finally:
        final_path = checkpoint_dir / f"ppo_final_{model.num_timesteps}.zip"
        model.save(final_path)
        print(f"[checkpoint] {final_path}")
        train_env.close()


if __name__ == "__main__":
    main()
