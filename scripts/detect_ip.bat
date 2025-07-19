@echo off
setlocal enabledelayedexpansion
:: DDOSai IP Detection Helper Script for Windows
:: This script helps you find your IP address for simulation testing

echo 🔍 DDOSai IP Detection Helper
echo =================================
echo.

:: Method 1: Using PowerShell with external service (same as the app)
echo 1. External IP (public internet IP):
echo    Using ipify.org (same service as DDOSai app):
powershell -Command "try { $ip = (Invoke-RestMethod -Uri 'https://api.ipify.org?format=json').ip; Write-Host '   ✅' $ip } catch { Write-Host '   ❌ Failed to get external IP' }"
echo.

:: Method 2: Local network IP
echo 2. Local network IP (for local testing):
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set "ip=%%a"
    set "ip=!ip: =!"
    if not "!ip!"=="127.0.0.1" (
        echo    ✅ !ip!
        goto :found_local
    )
)
echo    ❌ Could not detect local IP
:found_local

echo.

:: Method 3: All network interfaces
echo 3. All network interfaces:
echo    Available IPs:
ipconfig | findstr /c:"IPv4 Address" | for /f "tokens=2 delims=:" %%a in ('more') do (
    set "ip=%%a"
    set "ip=!ip: =!"
    echo    • !ip!
)

echo.
echo 💡 Recommendations:
echo    • For testing on the same machine: Use 127.0.0.1 (localhost)
echo    • For local network testing: Use your local IP (192.168.x.x or 10.x.x.x)
echo    • For internet testing: Use your external IP (⚠️  only if you own the target)
echo.
echo ⚠️  WARNING: Only attack systems you own or have explicit permission to test!
echo    Unauthorized DDoS attacks are illegal and can result in criminal charges.
echo.
pause
