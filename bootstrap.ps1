# bootstrap.ps1
$ErrorActionPreference = "Stop"
$proj = $PSScriptRoot; if (-not $proj) { $proj = (Get-Location).Path }
Set-Location $proj

if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

if (Test-Path "requirements.lock") {
  & .\.venv\Scripts\pip.exe install -r requirements.lock
} elseif (Test-Path "requirements.txt") {
  & .\.venv\Scripts\pip.exe install -r requirements.txt
}

New-Item -ItemType Directory -Force -Path "data","out","cache" | Out-Null

try {
  & .\.venv\Scripts\pip.exe install playwright | Out-Null
  & .\.venv\Scripts\python.exe -m playwright install | Out-Null
} catch { }

"BOOTSTRAP_OK"
