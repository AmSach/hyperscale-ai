@echo off
title SuperAI Upscaler GPU Studio Server
cd /d "%~dp0"

echo ================================================================
echo       Starting SuperAI Upscaler Local GPU Studio
echo ================================================================
echo.

if exist "venv\Scripts\python.exe" (
    echo [INFO] Activating virtual environment...
    set PYTHON_EXE=venv\Scripts\python.exe
) else (
    echo [INFO] Using system Python...
    set PYTHON_EXE=python
)

echo [INFO] Launching local web interface at http://localhost:8080...
start "" "http://localhost:8080"

%PYTHON_EXE% server.py
pause
