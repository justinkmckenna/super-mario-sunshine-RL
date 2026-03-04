from sms_rl.baselines import CenteringPolicy, RandomPolicy, run_episode
from sms_rl.envs.blooper_surfing import BlooperSurfingEnv


def test_env_reset_and_step_shapes() -> None:
    env = BlooperSurfingEnv()
    try:
        observation, info = env.reset()
        assert observation.shape == env.observation_space.shape
        assert "mock_progress" in info

        next_observation, reward, terminated, truncated, info = env.step(1)
        assert next_observation.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_components" in info
    finally:
        env.close()


def test_scripted_policy_beats_random_in_mock_env() -> None:
    scripted_rewards = []
    random_rewards = []

    env = BlooperSurfingEnv()
    try:
        for _ in range(5):
            scripted_rewards.append(run_episode(env, CenteringPolicy()).total_reward)
            random_rewards.append(run_episode(env, RandomPolicy(env.action_space.n)).total_reward)
    finally:
        env.close()

    assert sum(scripted_rewards) > sum(random_rewards)
