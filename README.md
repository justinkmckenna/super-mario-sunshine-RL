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
- controller input through `vgamepad` or keyboard mode
- frame capture through `DXcam`
- progress / finish / failure from `dolphin-memory-engine`

Install the extra dependencies with:

```bash
pip install -e .[windows-dolphin]
```

For PPO training and eval video export, install:

```bash
pip install -e .[training,windows-dolphin]
```

If PPO prints `Using cpu device` and you want GPU training on NVIDIA, install
CUDA-enabled PyTorch in the venv:

```bash
python -m pip uninstall -y torch
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected output should include `+cu124` and `True` for CUDA availability.

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
- Sunshine ISO: `C:\Users\justi\Downloads\Super Mario Sunshine (2002)(Nintendo)(US).iso`

For convenience, see `scripts/run_dolphin_smoke.ps1`.
For baseline comparison runs, use `scripts/run_dolphin_baselines.ps1`.

## Finding Memory Addresses

The current driver expects up to three memory bindings:

- progress value
- mission finished flag
- mission failed flag

Current working draft progress binding for USA Sunshine (`GMSE01`):

- `progress_address=0x80FA50D4`
- `progress_type=float`

Current working draft mission-failed binding for USA Sunshine (`GMSE01`):

- `failed_address=0x804257D3`
- `failed_type=byte`
- `failed_value=1`

Current working draft mission-success binding for USA Sunshine (`GMSE01`) on
purple-blooper start savestate:

- `finished_address=0x805F64C6`
- `finished_type=byte`
- `finished_value=1`

Current working savestate path:

- `C:\Users\justi\Downloads\purple-blooper-start.sav`

## Run Real Baselines

Smoke test (single neutral episode):

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\run_dolphin_smoke.ps1
```

Baseline comparison (5 neutral + 5 random episodes):

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\run_dolphin_baselines.ps1
```

## PPO Training

Run a first training pass with checkpointing + periodic eval:

```bash
python -m sms_rl.train_ppo ^
  --run-name ppo_blooper_v1 ^
  --total-timesteps 50000 ^
  --eval-every 5000 ^
  --eval-episodes 3 ^
  --checkpoint-every 5000 ^
  --dolphin-exe "C:\Users\justi\Downloads\dolphin-2512-x64\Dolphin-x64\Dolphin.exe" ^
  --game-path "C:\Users\justi\Downloads\Super Mario Sunshine (2002)(Nintendo)(US).iso" ^
  --save-state "C:\Users\justi\Downloads\purple-blooper-start.sav" ^
  --user-path "C:\Users\justi\Projects\super-mario-sunshine-RL\dolphin_user_profile" ^
  --window-title "Super Mario Sunshine" ^
  --control-mode vgamepad ^
  --capture-fps 30 ^
  --progress-address 0x80FA50D4 --progress-type float ^
  --finished-address 0x805F64C6 --finished-type byte --finished-value 1 ^
  --failed-address 0x804257D3 --failed-type byte --failed-value 1
```

To force GPU in training, add:

```bash
--device cuda
```

Outputs are written to `runs/<run-name>/`:

- `checkpoints/*.zip`
- `eval/eval_metrics.csv`
- `eval/*.mp4` (if `--record-eval-video`)
- TensorBoard logs in `tensorboard/`

PowerShell wrapper with current local paths:

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\run_dolphin_ppo_train.ps1
```

Current baseline scripts are configured for vgamepad control mode.

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

Use `scripts/find_success_flag_candidates.py` for success discovery with modes:

- `--mode basic`: safe/success/reset cycles
- `--mode win-only`: requires fail states to stay at baseline
- `--mode true-finish`: distinguishes bad terminals from true finish

## Integration Notes

The real Dolphin integration should implement the `BlooperDriver` protocol. That adapter is responsible for:

- restoring the savestate
- applying controller inputs
- capturing the latest frame
- reading progress / success / failure signals from memory or another reliable source

The Gymnasium environment intentionally keeps those concerns outside the policy-facing interface.
