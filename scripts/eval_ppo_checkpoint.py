from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sms_rl.train_ppo import build_env_factory, build_parser, evaluate_model


def build_eval_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Evaluate a saved PPO checkpoint on Blooper Surfing."
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--action-log", type=Path)
    return parser


def evaluate_model_with_action_log(
    model,
    env,
    eval_episodes: int,
    *,
    deterministic: bool,
    record_video: bool,
    video_path: Path | None,
    video_fps: int,
    grayscale: bool,
    action_log_path: Path,
) -> dict[str, float]:
    import imageio.v2 as imageio
    import torch
    from stable_baselines3.common.utils import obs_as_tensor

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    success_count = 0
    fail_count = 0
    captured_frames: list[np.ndarray] = []

    action_log_path.parent.mkdir(parents=True, exist_ok=True)
    with action_log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode",
                "step",
                "deterministic",
                "chosen_action",
                "prob_left",
                "prob_neutral",
                "prob_right",
                "prob_jump",
                "reward",
                "progress",
                "raw_progress",
                "path_progress",
                "path_progress_raw",
                "path_distance",
                "path_segment_index",
                "path_distance_to_start",
                "path_progress_regression_clamped",
                "path_respawn_detected",
                "reward_progress",
                "reward_path_distance",
                "reward_step_penalty",
                "reward_finish",
                "reward_fail",
                "episode_elapsed_seconds",
                "mission_finished_raw",
                "mission_failed_raw",
                "mission_finished",
                "mission_failed",
                "mission_finished_confirm_count",
                "mission_failed_confirm_count",
                "terminated",
                "truncated",
            ]
        )

        for episode_idx in range(eval_episodes):
            obs, info = env.reset()
            del info
            terminated = False
            truncated = False
            ep_reward = 0.0
            ep_steps = 0
            if record_video and episode_idx == 0:
                captured_frames.append(np.array(_obs_to_frame(obs, grayscale), copy=True))

            while not terminated and not truncated:
                batched_obs = np.expand_dims(obs, axis=0)
                obs_tensor = obs_as_tensor(batched_obs, model.device)
                with torch.no_grad():
                    distribution = model.policy.get_distribution(obs_tensor)
                    probs = distribution.distribution.probs.detach().cpu().numpy()[0]
                action, _ = model.predict(obs, deterministic=deterministic)
                chosen_action = int(action)
                obs, reward, terminated, truncated, info = env.step(chosen_action)
                ep_reward += float(reward)
                reward_components = info.get("reward_components", {})
                writer.writerow(
                    [
                        episode_idx,
                        ep_steps,
                        deterministic,
                        chosen_action,
                        float(probs[0]),
                        float(probs[1]),
                        float(probs[2]),
                        float(probs[3]),
                        float(reward),
                        float(info.get("progress", 0.0)),
                        float(info.get("raw_progress", 0.0)),
                        float(info.get("path_progress", info.get("progress", 0.0))),
                        float(info.get("path_progress_raw", 0.0)),
                        float(info.get("path_distance", 0.0)),
                        float(info.get("path_segment_index", 0.0)),
                        float(info.get("path_distance_to_start", 0.0)),
                        float(info.get("path_progress_regression_clamped", 0.0)),
                        float(info.get("path_respawn_detected", 0.0)),
                        float(reward_components.get("progress", 0.0)),
                        float(reward_components.get("path_distance", 0.0)),
                        float(reward_components.get("step_penalty", 0.0)),
                        float(reward_components.get("finish", 0.0)),
                        float(reward_components.get("fail", 0.0)),
                        float(info.get("episode_elapsed_seconds", 0.0)),
                        bool(info.get("mission_finished_raw", False)),
                        bool(info.get("mission_failed_raw", False)),
                        bool(info.get("mission_finished", False)),
                        bool(info.get("mission_failed", False)),
                        int(info.get("mission_finished_confirm_count", 0)),
                        int(info.get("mission_failed_confirm_count", 0)),
                        bool(terminated),
                        bool(truncated),
                    ]
                )
                ep_steps += 1
                if record_video and episode_idx == 0:
                    captured_frames.append(np.array(_obs_to_frame(obs, grayscale), copy=True))

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_steps)
            success_count += int(bool(info.get("mission_finished", False)))
            fail_count += int(bool(info.get("mission_failed", False)))

    if record_video and video_path is not None and captured_frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
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


def _obs_to_frame(observation: np.ndarray, grayscale: bool) -> np.ndarray:
    if grayscale:
        last = observation[-1]
        return np.repeat(last[:, :, None], 3, axis=2)
    channels_per_frame = 3
    frame = observation[-channels_per_frame:, :, :]
    return np.transpose(frame, (1, 2, 0))


def main() -> None:
    parser = build_eval_parser()
    args = parser.parse_args()

    from stable_baselines3 import PPO

    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    make_env = build_env_factory(args)
    env = make_env()
    try:
        model = PPO.load(args.checkpoint_path, device=args.device)
        if args.action_log is None:
            metrics = evaluate_model(
                model,
                env,
                args.eval_episodes,
                deterministic=args.deterministic,
                record_video=args.record_eval_video,
                video_path=args.output_video,
                video_fps=args.eval_video_fps,
                grayscale=args.grayscale,
            )
        else:
            metrics = evaluate_model_with_action_log(
                model,
                env,
                args.eval_episodes,
                deterministic=args.deterministic,
                record_video=args.record_eval_video,
                video_path=args.output_video,
                video_fps=args.eval_video_fps,
                grayscale=args.grayscale,
                action_log_path=args.action_log,
            )
    finally:
        env.close()

    print(
        "[eval] "
        f"checkpoint={args.checkpoint_path} "
        f"mean_reward={metrics['mean_reward']:.3f} "
        f"mean_length={metrics['mean_length']:.2f} "
        f"success_rate={metrics['success_rate']:.2%} "
        f"fail_rate={metrics['fail_rate']:.2%}"
    )
    if args.record_eval_video and args.output_video is not None:
        print(f"[video] {args.output_video}")
    if args.action_log is not None:
        print(f"[actions] {args.action_log}")


if __name__ == "__main__":
    main()
