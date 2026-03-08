$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$dolphinExe = "C:\Users\justi\Downloads\dolphin-2512-x64\Dolphin-x64\Dolphin.exe"
$gamePath = "C:\Users\justi\Downloads\Super Mario Sunshine (2002)(Nintendo)(US).iso"
$saveStatePath = "C:\Users\justi\Downloads\purple-blooper-start.sav"
$userPath = "C:\Users\justi\Projects\super-mario-sunshine-RL\dolphin_user_profile"
$windowTitle = "Super Mario Sunshine"

# Locked v0 bindings (purple blooper start).
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

$commonArgs = @(
  "-m", "sms_rl.cli",
  "--backend", "dolphin",
  "--control-mode", "vgamepad",
  "--window-title", $windowTitle,
  "--episodes", "3",
  "--capture-fps", "30",
  "--dolphin-exe", $dolphinExe,
  "--game-path", $gamePath,
  "--save-state", $saveStatePath,
  "--user-path", $userPath,
  "--progress-address", $progressAddress,
  "--progress-type", "float",
  "--finished-address", $finishedAddress,
  "--finished-type", "byte",
  "--finished-value", "1",
  "--failed-address", $failedAddress,
  "--failed-type", "byte",
  "--failed-value", "1"
)

try {
  $awakeFlags = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED)
  [Win32.NativeMethods]::SetThreadExecutionState($awakeFlags) | Out-Null

  Write-Host "Running neutral baseline..."
  & $python @commonArgs --baseline neutral

  Write-Host ""
  Write-Host "Running random baseline..."
  & $python @commonArgs --baseline random
}
finally {
  [Win32.NativeMethods]::SetThreadExecutionState([uint32]$ES_CONTINUOUS) | Out-Null
}
