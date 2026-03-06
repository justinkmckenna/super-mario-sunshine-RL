# Super Mario Sunshine RL

This repository contains the first environment scaffold for a constrained Super Mario Sunshine reinforcement learning task.

Current scope:

- Gymnasium-compatible environment
- Windows-first design
- raw-pixel observations for the policy
- memory-backed reward and termination signals allowed
- mock backend for local smoke tests before Dolphin integration
- first Dolphin driver scaffold for Windows

## Layout

- `sms_rl/envs/blooper_surfing.py`: Gymnasium environment
- `sms_rl/drivers/base.py`: integration protocol for emulator control and state capture
- `sms_rl/drivers/mock.py`: deterministic mock driver for smoke testing
- `sms_rl/drivers/dolphin.py`: Windows Dolphin driver using virtual gamepad, DXcam, and Dolphin Memory Engine
- `sms_rl/baselines.py`: random and scripted baseline helpers
- `sms_rl/cli.py`: simple baseline runner entrypoint

## Smoke Test

Install the package in editable mode and run a mock rollout:

```bash
pip install -e .
sms-rl --episodes 3 --baseline scripted
```

That validates the environment loop without requiring Dolphin.

You can also call the installed script directly:

```bash
sms-rl --episodes 3 --baseline scripted
```

## Dolphin Integration

The first real driver scaffold is available in `sms_rl/drivers/dolphin.py`.

It is designed around:

- Dolphin launched on Windows in windowed mode
- controller input through `vgamepad`
- frame capture through `DXcam`
- progress / finish / failure from `dolphin-memory-engine`

Install the extra dependencies with:

```bash
pip install -e .[windows-dolphin]
```

The driver still requires project-specific configuration before it can run:

- Dolphin executable path
- game path
- savestate path
- Dolphin window title match
- memory addresses for progress, finish, and fail signals

Example CLI shape for a first real smoke test:

```bash
sms-rl --backend dolphin --baseline neutral --episodes 1 \
  --dolphin-exe "C:\Path\To\Dolphin.exe" \
  --game-path "C:\Path\To\Super Mario Sunshine.iso" \
  --save-state "C:\Path\To\blooper_start.s01" \
  --progress-address 0x12345678 \
  --finished-address 0x12345690 \
  --failed-address 0x123456A0
```

That command is only usable after you identify working Sunshine memory addresses.

Current local paths on this machine:

- Dolphin executable: `C:\Users\justi\Downloads\dolphin-2512-x64\Dolphin-x64\Dolphin.exe`
- Sunshine ISO: `C:\Users\justi\Downloads\Super Mario Sunshine (2002)(Nintendo)(EU)(M5).iso`

For convenience, see `scripts/run_dolphin_smoke.ps1`.

## Finding Memory Addresses

The current driver expects up to three memory bindings:

- progress value
- mission finished flag
- mission failed flag

Current working draft progress binding for USA Sunshine (`GMSE01`):

- `progress_address=0x80FA50D4`
- `progress_type=float`

The practical workflow is:

1. Launch Dolphin and the game manually.
2. Connect a memory inspection tool to the running Dolphin process.
3. Find values that change predictably during Blooper Surfing.
4. Confirm the same addresses still work after a savestate reload.

For a first pass, likely useful targets are:

- race progress or checkpoint progress
- a mission state byte that changes on success
- a state byte that changes on wipeout or mission failure

Once you identify candidate addresses, plug them into the smoke-test script or CLI flags.

## Integration Notes

The real Dolphin integration should implement the `BlooperDriver` protocol. That adapter is responsible for:

- restoring the savestate
- applying controller inputs
- capturing the latest frame
- reading progress / success / failure signals from memory or another reliable source

The Gymnasium environment intentionally keeps those concerns outside the policy-facing interface.
