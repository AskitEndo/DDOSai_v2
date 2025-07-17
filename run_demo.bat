@echo off
echo ===================================================
echo DDoS.AI Platform - Demo Mode
echo ===================================================
echo.
echo This script will start the DDoS.AI platform in demo mode
echo with pre-configured settings and sample data.
echo.
echo Components that will be started:
echo  - Backend API (FastAPI)
echo  - Frontend UI (React)
echo  - Prometheus (Metrics)
echo  - Grafana (Dashboards)
echo.
echo Press Ctrl+C to stop all services.
echo.
echo ===================================================
echo.

REM Check if Docker is installed
docker --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not installed or not in PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker Compose is not installed or not in PATH.
    echo Docker Compose is included with Docker Desktop for Windows.
    exit /b 1
)

echo Setting up environment for demo mode...
set DEMO_MODE=true
set LOAD_SAMPLE_DATA=true
set ENABLE_SIMULATION=true
set LOG_LEVEL=INFO
set PROMETHEUS_ENABLED=true

echo Starting DDoS.AI platform in demo mode...
docker-compose -f docker-compose.yml up -d

echo.
echo Waiting for services to start...
timeout /t 10 /nobreak > nul

REM Check if services are running
docker-compose ps
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to start services.
    exit /b 1
)

echo.
echo ===================================================
echo DDoS.AI Platform is now running in demo mode!
echo.
echo Access the dashboard at: http://localhost:3000
echo API documentation at: http://localhost:8000/docs
echo Grafana dashboards at: http://localhost:3001
echo.
echo Sample credentials:
echo  - Username: demo
echo  - Password: demo123
echo.
echo Running pre-configured demo scenario...
echo.

REM Start demo scenario in background
start /b python -m backend.demo.run_scenario --scenario=syn_flood

echo.
echo Demo scenario started. You should see attack detection events
echo in the dashboard within 30 seconds.
echo.
echo Press any key to stop all services...
pause > nul

echo.
echo Stopping all services...
docker-compose down

echo.
echo DDoS.AI Platform demo stopped.
echo Thank you for trying DDoS.AI!
echo.