$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$checkpointPath = "C:\Users\justi\Projects\super-mario-sunshine-RL\runs\ppo_overnight_pathreward_v2_continue\checkpoints\ppo_step_300000.zip"
$outputVideo = "C:\Users\justi\Projects\super-mario-sunshine-RL\runs\ppo_overnight_pathreward_v2_continue\eval\manual_eval_300000.mp4"
$actionLog = "C:\Users\justi\Projects\super-mario-sunshine-RL\runs\ppo_overnight_pathreward_v2_continue\eval\manual_eval_300000_actions.csv"

$dolphinExe = "C:\Users\justi\Downloads\dolphin-2512-x64\Dolphin-x64\Dolphin.exe"
$gamePath = "C:\Users\justi\Downloads\Super Mario Sunshine (2002)(Nintendo)(US).iso"
$saveStatePath = "C:\Users\justi\Downloads\behind-purple-blooper-start.sav"
$userPath = "C:\Users\justi\Projects\super-mario-sunshine-RL\dolphin_user_profile"
$windowTitle = "Super Mario Sunshine"
$progressAddress = "0x80FA50D4"
$finishedAddress = "0x805F64C6"
$failedAddress = "0x804257D3"
$progressRewardScale = "0.001"
$pathDistancePenaltyScale = "0.0005"

& $python scripts\eval_ppo_checkpoint.py `
  --checkpoint-path $checkpointPath `
  --eval-episodes 3 `
  --output-video $outputVideo `
  --action-log $actionLog `
  --device cuda `
  --n-steps 128 `
  --action-repeat 2 `
  --max-episode-seconds 45 `
  --progress-reward-scale $progressRewardScale `
  --path-distance-penalty-scale $pathDistancePenaltyScale `
  --dolphin-exe $dolphinExe `
  --game-path $gamePath `
  --save-state $saveStatePath `
  --user-path $userPath `
  --window-title $windowTitle `
  --no-dolphin-batch-mode `
  --render-to-main `
  --control-mode vgamepad `
  --capture-backend mss `
  --capture-fps 30 `
  --post-launch-delay-seconds 0 `
  --post-reset-delay-seconds 0 `
  --startup-forward-seconds 1.0 `
  --startup-forward-magnitude 1.0 `
  --startup-settle-seconds 0.1 `
  --window-stable-seconds 0 `
  --progress-address $progressAddress `
  --progress-type float `
  --finished-address $finishedAddress `
  --finished-type byte `
  --finished-value 1 `
  --failed-address $failedAddress `
  --failed-type byte `
  --failed-value 1
