$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$dolphinExe = "C:\Users\justi\Downloads\dolphin-2512-x64\Dolphin-x64\Dolphin.exe"
$gamePath = "C:\Users\justi\Downloads\Super Mario Sunshine (2002)(Nintendo)(EU)(M5).iso"

# Fill these in once the Sunshine memory map is known.
$progressAddress = "0x00000000"
$finishedAddress = "0x00000000"
$failedAddress = "0x00000000"

& $python -m sms_rl.cli `
  --backend dolphin `
  --baseline neutral `
  --episodes 1 `
  --dolphin-exe $dolphinExe `
  --game-path $gamePath `
  --progress-address $progressAddress `
  --finished-address $finishedAddress `
  --failed-address $failedAddress
