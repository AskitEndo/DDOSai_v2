@echo off
echo Starting DDoS.AI Platform in Development Mode...

REM Check if Docker is running
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not running. Please start Docker and try again.
    exit /b 1
)

REM Build and start the containers in development mode
echo Building and starting containers in development mode...
docker-compose -f docker-compose.dev.yml up -d --build

echo.
echo DDoS.AI Platform is now running in development mode!
echo.
echo Access the services at:
echo - Frontend: http://localhost:3000
echo - Backend API: http://localhost:8000
echo.
echo The code is mounted as volumes, so changes will trigger hot-reloading.
echo.
echo To stop the platform, run: docker-compose -f docker-compose.dev.yml down
echo.