$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$outputCsv = "C:\Users\justi\Downloads\manual_race_trace.csv"
$progressAddress = "0x80FA50D4"
$finishedAddress = "0x805F64C6"
$failedAddress = "0x804257D3"

& $python scripts\log_manual_race_trace.py `
  --progress-address $progressAddress `
  --finished-address $finishedAddress `
  --failed-address $failedAddress `
  --output-csv $outputCsv `
  --interval-seconds 0.1 `
  --max-seconds 90
