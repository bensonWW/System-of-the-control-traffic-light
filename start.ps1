$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " TrafficVision - Local Startup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/2] Starting FastAPI backend (port 8000)..." -ForegroundColor Yellow
$apiScript = Join-Path $root "TrafficVision Design System\serve_api.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python '$apiScript'"

Start-Sleep -Seconds 3

Write-Host "[2/2] Starting runtime pipeline (every 5 min)..." -ForegroundColor Yellow
$toolsDir = Join-Path $root "tools"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$toolsDir'; python runtime_pipeline.py"

Write-Host ""
Write-Host "Two windows opened:" -ForegroundColor Green
Write-Host "  API:      http://localhost:8000/api/status" -ForegroundColor Green
Write-Host "  Pipeline: runs every 5 minutes" -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard: open TrafficVision Design System\ui_kits\traffic-dashboard\dashboard.html"
Write-Host ""
Write-Host "Press Enter to close this window..."
Read-Host
