param([string]$OutDir="out")
$run = Get-ChildItem $OutDir | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $run) { Write-Output "DELIVERY_INCOMPLETE missing out dir"; exit 1 }
$zip = Join-Path $OutDir "$($run.Name).zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $run.FullName "*") -DestinationPath $zip
$sha = Get-FileHash $zip -Algorithm SHA256 | Select-Object -ExpandProperty Hash
Write-Output "PACK_OK $zip $sha"