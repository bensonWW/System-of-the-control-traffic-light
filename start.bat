@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  TrafficVision - Local Startup
echo ============================================
echo.
echo [1/2] Starting FastAPI backend (port 8000)...
start "TrafficVision API" cmd /k python "%~dp0TrafficVision Design System\serve_api.py"

timeout /t 3 /nobreak >nul

echo [2/2] Starting runtime pipeline (every 5 min)...
start "TrafficVision Pipeline" cmd /k "cd /d %~dp0tools && python runtime_pipeline.py"

echo.
echo Two windows opened:
echo   - TrafficVision API      http://localhost:8000/api/status
echo   - TrafficVision Pipeline runs every 5 minutes
echo.
echo Dashboard: open TrafficVision Design System\ui_kits\traffic-dashboard\dashboard.html
pause
