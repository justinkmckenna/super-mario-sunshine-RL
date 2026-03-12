param(
  [string]$RunDir = ".\runs\ppo_blooper_v1",
  [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"

$statusPath = Join-Path $RunDir "status.json"
$logPath = Join-Path $RunDir "train.log"

Write-Host "Watching training status..."
Write-Host "Run dir: $RunDir"
Write-Host "Status file: $statusPath"
Write-Host "Poll interval: ${PollSeconds}s"

while ($true) {
  $pythonProc = Get-Process python -ErrorAction SilentlyContinue
  $dolphinProc = Get-Process Dolphin -ErrorAction SilentlyContinue

  $logLastWrite = $null
  if (Test-Path $logPath) {
    $logLastWrite = (Get-Item $logPath).LastWriteTime.ToString("o")
  }

  $status = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    python_alive = [bool]$pythonProc
    dolphin_alive = [bool]$dolphinProc
    log_last_write = $logLastWrite
  }

  $status | ConvertTo-Json | Set-Content -Path $statusPath -Encoding UTF8
  Write-Host ("[{0}] python={1} dolphin={2} log={3}" -f $status.timestamp, $status.python_alive, $status.dolphin_alive, $status.log_last_write)

  Start-Sleep -Seconds $PollSeconds
}
