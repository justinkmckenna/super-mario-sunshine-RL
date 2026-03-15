$ErrorActionPreference = "Stop"

& ".\.venv\Scripts\python" "scripts\dump_env_frames.py" `
  --output-dir "C:\Users\justi\Downloads\frame_dump" `
  --steps 5 `
  --dolphin-exe "C:\Users\justi\Downloads\dolphin-2512-x64\Dolphin-x64\Dolphin.exe" `
  --game-path "C:\Users\justi\Downloads\Super Mario Sunshine (2002)(Nintendo)(US).iso" `
  --save-state "C:\Users\justi\Downloads\behind-purple-blooper-start.sav" `
  --user-path "C:\Users\justi\Projects\super-mario-sunshine-RL\dolphin_user_profile" `
  --window-title "Super Mario Sunshine" `
  --render-to-main `
  --control-mode vgamepad `
  --capture-backend mss `
  --capture-fps 30 `
  --post-launch-delay-seconds 0 `
  --post-reset-delay-seconds 0. `
  --startup-forward-seconds 1.0 `
  --startup-forward-magnitude 1.0 `
  --startup-settle-seconds 0.1 `
  --window-stable-seconds 0 `
  --no-dolphin-batch-mode `
  --action-repeat 2 `
  --progress-address 0x80FA50D4 `
  --progress-type float `
  --finished-address 0x805F64C6 `
  --finished-type byte `
  --finished-value 1 `
  --failed-address 0x804257D3 `
  --failed-type byte `
  --failed-value 1
