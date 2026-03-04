from __future__ import annotations

import argparse

from sms_rl.baselines import CenteringPolicy, RandomPolicy, run_episode
from sms_rl.config import EnvConfig
from sms_rl.envs.blooper_surfing import BlooperSurfingEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Blooper Surfing smoke tests.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--baseline",
        choices=("random", "scripted"),
        default="scripted",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    env = BlooperSurfingEnv(config=EnvConfig())
    try:
        if args.baseline == "random":
            policy = RandomPolicy(env.action_space.n, seed=0)
        else:
            policy = CenteringPolicy()

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
