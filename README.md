# Super Mario Sunshine RL

This repository contains the first environment scaffold for a constrained Super Mario Sunshine reinforcement learning task.

Current scope:

- Gymnasium-compatible environment
- Windows-first design
- raw-pixel observations for the policy
- memory-backed reward and termination signals allowed
- mock backend for local smoke tests before Dolphin integration

## Layout

- `sms_rl/envs/blooper_surfing.py`: Gymnasium environment
- `sms_rl/drivers/base.py`: integration protocol for emulator control and state capture
- `sms_rl/drivers/mock.py`: deterministic mock driver for smoke testing
- `sms_rl/baselines.py`: random and scripted baseline helpers
- `sms_rl/cli.py`: simple baseline runner entrypoint

## Smoke Test

Install the package in editable mode and run a mock rollout:

```bash
pip install -e .
sms-rl --episodes 3 --baseline scripted
```

That validates the environment loop without requiring Dolphin.

## Integration Notes

The real Dolphin integration should implement the `BlooperDriver` protocol. That adapter is responsible for:

- restoring the savestate
- applying controller inputs
- capturing the latest frame
- reading progress / success / failure signals from memory or another reliable source

The Gymnasium environment intentionally keeps those concerns outside the policy-facing interface.
