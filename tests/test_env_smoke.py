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


def test_baseline_episodes_run_in_mock_env() -> None:
    env = BlooperSurfingEnv()
    try:
        scripted = run_episode(env, CenteringPolicy())
        random_result = run_episode(env, RandomPolicy(env.action_space.n, seed=0))
    finally:
        env.close()

    assert scripted.steps > 0
    assert random_result.steps > 0
    assert scripted.terminated or scripted.truncated
    assert random_result.terminated or random_result.truncated
