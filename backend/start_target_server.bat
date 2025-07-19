@echo off
setlocal enabledelayedexpansion

echo 🎯 DDoS Simulation Target Server Launcher
echo ==========================================
echo.

cd /d "%~dp0"

echo 📂 Current directory: %CD%
echo.

echo 🔍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    echo.
    pause
    exit /b 1
)

python --version
echo ✅ Python is available
echo.

echo 🚀 Starting DDoS Target Server...
echo 📊 This server will show you REAL attack traffic!
echo 💡 Use this IP:PORT in your simulation: 127.0.0.1:8080
echo.
echo ⚡ LIVE MONITORING: You will see requests coming in real-time
echo 🔄 Leave this window open during simulation testing
echo.
echo Press Ctrl+C to stop the server
echo ==========================================
echo.

python ddos_target_server.py --port 8080

echo.
echo 🛑 Server stopped.
pause
