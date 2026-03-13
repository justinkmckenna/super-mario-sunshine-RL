$ErrorActionPreference = "Stop"

& ".\.venv\Scripts\python" "scripts\action_timing_probe.py" `
  --policy scripted-midtest `
  --episodes 1 `
  --probe-seconds 6 `
  --max-decisions 80 `
  --output-csv "C:\Users\justi\Downloads\action_timing_midtest.csv" `
  --dolphin-exe "C:\Users\justi\Downloads\dolphin-2512-x64\Dolphin-x64\Dolphin.exe" `
  --game-path "C:\Users\justi\Downloads\Super Mario Sunshine (2002)(Nintendo)(US).iso" `
  --save-state "C:\Users\justi\Downloads\purple-blooper-start.sav" `
  --user-path "C:\Users\justi\Projects\super-mario-sunshine-RL\dolphin_user_profile" `
  --window-title "Super Mario Sunshine" `
  --control-mode vgamepad `
  --capture-fps 30 `
  --log-every-step `
  --no-dolphin-batch-mode `
  --no-restart-on-reset `
  --save-state-slot 1 `
  --action-repeat 2 `
  --progress-address 0x80FA50D4 `
  --progress-type float `
  --finished-address 0x805F64C6 `
  --finished-type byte `
  --finished-value 1 `
  --failed-address 0x804257D3 `
  --failed-type byte `
  --failed-value 1
