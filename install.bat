<# :
@echo off
title SuperAI Upscaler - Complete GPU Studio Setup
cd /d "%~dp0"
echo ================================================================
echo   Starting SuperAI Upscaler Professional GPU Setup...
echo ================================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-Command -ScriptBlock ([ScriptBlock]::Create([IO.File]::ReadAllText('%~f0')))"
pause
goto :EOF
#>

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor 3072

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "     SuperAI Upscaler — Professional GPU Studio Setup           " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$installDir = $PSScriptRoot
if (-not (Test-Path "$installDir\server.py")) {
    $installDir = "$env:USERPROFILE\HyperScale-AI"
    if (-not (Test-Path $installDir)) {
        New-Item -ItemType Directory -Force -Path $installDir | Out-Null
    }
}
Set-Location -Path $installDir

# 1. System GPU Check
Write-Host "[1/6] Checking Hardware & GPU..." -ForegroundColor Yellow
try {
    $gpuInfo = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" }
    if ($gpuInfo) {
        Write-Host "  -> Found NVIDIA GPU: $($gpuInfo[0].Name)" -ForegroundColor Green
    } else {
        Write-Host "  -> [WARNING] No NVIDIA GPU detected! The app will install and run on CPU mode." -ForegroundColor DarkYellow
    }
} catch {
    Write-Host "  -> Skipping detailed GPU check." -ForegroundColor Gray
}

# 2. Check or Install Python
Write-Host "`n[2/6] Checking Python Environment..." -ForegroundColor Yellow
$pythonExe = $null

try {
    $verOut = python --version 2>&1
    if ($verOut -match "Python 3\.(1[0-2])") {
        $pythonExe = (Get-Command python).Source
        Write-Host "  -> Found suitable Python: $verOut ($pythonExe)" -ForegroundColor Green
    }
} catch {}

if (-not $pythonExe) {
    $candidates = @(
        "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe",
        "$env:USERPROFILE\AppData\Local\Programs\Python\Python311\python.exe",
        "$env:USERPROFILE\AppData\Local\Programs\Python\Python310\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe"
    )
    foreach ($cand in $candidates) {
        if (Test-Path $cand) {
            $pythonExe = $cand
            Write-Host "  -> Found existing Python: $pythonExe" -ForegroundColor Green
            break
        }
    }
}

if (-not $pythonExe) {
    Write-Host "  -> Python 3.10-3.12 not found. Downloading Python 3.12..." -ForegroundColor Yellow
    $pyInst = "$env:TEMP\python-installer.exe"
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe" -OutFile $pyInst -UseBasicParsing
    
    Write-Host "  -> Installing Python 3.12 silently..." -ForegroundColor Yellow
    Start-Process -FilePath $pyInst -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0" -Wait
    Remove-Item -Path $pyInst -Force -ErrorAction SilentlyContinue
    
    $pythonExe = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
    if (-not (Test-Path $pythonExe)) {
        Write-Host "[ERROR] Failed to install Python automatically. Please install Python 3.12 manually." -ForegroundColor Red
        exit 1
    }
    Write-Host "  -> Python installed successfully!" -ForegroundColor Green
}

# Ensure studio files exist in $installDir
if (-not (Test-Path "$installDir\server.py")) {
    Write-Host "  -> Downloading studio code package using Python..." -ForegroundColor Yellow
    & $pythonExe -c "import urllib.request, zipfile, io; r = urllib.request.urlopen('https://hyperscale-ai-amsachs-projects.vercel.app/app.zip'); z = zipfile.ZipFile(io.BytesIO(r.read())); z.extractall('$($installDir.Replace('\', '/'))')"
}

# 3. Setup Virtual Environment
Write-Host "`n[3/6] Setting up Virtual Environment..." -ForegroundColor Yellow
if (-not (Test-Path "$installDir\venv\Scripts\python.exe")) {
    Write-Host "  -> Creating virtual environment venv..." -ForegroundColor Yellow
    & $pythonExe -m venv venv
}
$venvPy = "$installDir\venv\Scripts\python.exe"
Write-Host "  -> Upgrading pip package manager..." -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip --quiet

# 4. Install PyTorch & AI Frameworks
Write-Host "`n[4/6] Installing PyTorch, HAT, & AI Libraries..." -ForegroundColor Yellow
if (Test-Path "$installDir\check_and_install_deps.py") {
    & $venvPy check_and_install_deps.py
} else {
    & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    & $venvPy -m pip install numpy pillow opencv-python spandrel basicsr gfpgan realesrgan diffusers transformers accelerate requests scikit-image
}

# 5. Pre-Download ALL AI Models (HAT + GFPGAN)
Write-Host "`n[5/6] Pre-Downloading ALL AI Models & Weights..." -ForegroundColor Yellow
if (Test-Path "$installDir\download_all_models.py") {
    & $venvPy download_all_models.py
}

# Ensure run_server.bat exists
$batPath = "$installDir\run_server.bat"
$batContent = "@echo off`r`ntitle SuperAI Upscaler GPU Studio Server`r`ncd /d `"%~dp0`"`r`necho ================================================================`r`necho       Starting SuperAI Upscaler Local GPU Studio`r`necho ================================================================`r`necho.`r`nif exist `"venv\Scripts\python.exe`" (`r`n    set PYTHON_EXE=venv\Scripts\python.exe`r`n) else (`r`n    set PYTHON_EXE=python`r`n)`r`nstart `"`" `"http://localhost:8080`"`r`n%PYTHON_EXE% server.py`r`npause`r`n"
[System.IO.File]::WriteAllText($batPath, $batContent)

# 6. Create Shortcut & Launch Studio
Write-Host "`n[6/6] Creating Desktop Shortcut..." -ForegroundColor Yellow
try {
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\SuperAI Upscaler Studio.lnk")
    $Shortcut.TargetPath = "$batPath"
    $Shortcut.WorkingDirectory = "$installDir"
    $Shortcut.Description = "SuperAI Upscaler Professional GPU Studio"
    $Shortcut.IconLocation = "shell32.dll,141"
    $Shortcut.Save()
    Write-Host "  -> Created Desktop Shortcut: SuperAI Upscaler Studio" -ForegroundColor Green
} catch {
    Write-Host "  -> Note: Shortcut creation skipped." -ForegroundColor Gray
}

Write-Host "`n================================================================" -ForegroundColor Green
Write-Host "   ALL MODELS INSTALLED & VERIFIED! STARTING GPU STUDIO..." -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "`nLaunching local web server and opening http://localhost:8080..." -ForegroundColor Cyan

Start-Process -FilePath "$batPath" -WorkingDirectory "$installDir"
