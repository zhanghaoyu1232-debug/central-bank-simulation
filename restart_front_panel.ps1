$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "Stopping old front-panel servers on ports 8000 and 8010..."
$connections = @()
$connections += Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
$connections += Get-NetTCPConnection -LocalPort 8010 -ErrorAction SilentlyContinue
$processIds = @($connections | Select-Object -ExpandProperty OwningProcess -Unique)

foreach ($processId in $processIds) {
    if ($processId -and $processId -ne $PID) {
        Write-Host "Stopping process $processId"
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Starting latest root server.py..."
Write-Host "Expected API version: scenario-front-panel-port-8010-v16"
Write-Host "Open: http://127.0.0.1:8010"
Write-Host "Health check: http://127.0.0.1:8010/api/health"

if (Get-Command python -ErrorAction SilentlyContinue) {
    python "$Root\server.py"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    py "$Root\server.py"
} else {
    throw "Python was not found. Install Python or add it to PATH."
}
