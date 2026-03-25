$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$dolphinExe = "C:\Users\justi\Downloads\dolphin-2512-x64\Dolphin-x64\Dolphin.exe"
$gamePath = "C:\Users\justi\Downloads\Super Mario Sunshine (2002)(Nintendo)(US).iso"
$saveStatePath = "C:\Users\justi\Downloads\behind-purple-blooper-start.sav"
$userPath = "C:\Users\justi\Projects\super-mario-sunshine-RL\dolphin_user_profile"
$windowTitle = "Super Mario Sunshine"

$progressAddress = "0x80FA50D4"
$finishedAddress = "0x805F64C6"
$failedAddress = "0x804257D3"

if (-not ("Win32.NativeMethods" -as [type])) {
  Add-Type -Namespace Win32 -Name NativeMethods -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
}

$ES_CONTINUOUS = [Convert]::ToUInt32("80000000", 16)
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$ES_DISPLAY_REQUIRED = [uint32]0x00000002

$runName = "ppo_overnight_pathreward_v2_continue"
$totalTimesteps = "300000"
$evalEvery = "10000"
$checkpointEvery = "10000"
$evalEpisodes = "3"
$nSteps = "128"
$actionRepeat = "2"
$learningRate = "0.0001"
$progressRewardScale = "0.001"
$pathDistancePenaltyScale = "0.0005"
$resumeCheckpoint = "C:\Users\justi\Projects\super-mario-sunshine-RL\runs\ppo_overnight_pathreward_v1\checkpoints\ppo_step_100000.zip"

try {
  $awakeFlags = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED)
  [Win32.NativeMethods]::SetThreadExecutionState($awakeFlags) | Out-Null

  & $python -m sms_rl.train_ppo `
    --run-name $runName `
    --device cuda `
    --learning-rate $learningRate `
    --n-steps $nSteps `
    --total-timesteps $totalTimesteps `
    --eval-every $evalEvery `
    --eval-episodes $evalEpisodes `
    --checkpoint-every $checkpointEvery `
    --action-repeat $actionRepeat `
    --max-episode-seconds 45 `
    --progress-reward-scale $progressRewardScale `
    --path-distance-penalty-scale $pathDistancePenaltyScale `
    --resume-checkpoint $resumeCheckpoint `
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
}
finally {
  [Win32.NativeMethods]::SetThreadExecutionState([uint32]$ES_CONTINUOUS) | Out-Null
}
