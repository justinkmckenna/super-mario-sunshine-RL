$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$dolphinExe = "C:\Users\justi\Downloads\dolphin-2512-x64\Dolphin-x64\Dolphin.exe"
$gamePath = "C:\Users\justi\Downloads\Super Mario Sunshine (2002)(Nintendo)(US).iso"

# Working draft progress address (USA Sunshine / GMSE01).
$progressAddress = "0x80FA50D4"
$finishedAddress = "0x00000000"
# Working draft failed flag address (USA Sunshine / GMSE01).
$failedAddress = "0x804257D3"

& $python -m sms_rl.cli `
  --backend dolphin `
  --baseline neutral `
  --episodes 1 `
  --dolphin-exe $dolphinExe `
  --game-path $gamePath `
  --progress-address $progressAddress `
  --finished-address $finishedAddress `
  --failed-address $failedAddress `
  --failed-type byte `
  --failed-value 1
