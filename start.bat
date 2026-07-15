@echo off
title AI Legal Assistant

echo ========================================
echo    AI Legal Assistant - Quick Start
echo ========================================
echo.

echo [check] python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] python not found in PATH
    echo        Try: py -3 main.py
    pause
    exit /b 1
)
echo [OK] python found
echo.

if not exist .venv\Scripts\activate.bat (
    echo [1/4] creating venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [FAIL] venv creation failed
        pause
        exit /b 1
    )
)
call .venv\Scripts\activate.bat
echo [1/4] venv ready

echo [2/4] checking dependencies...
pip install -r requirements.txt -q 2>nul
echo [2/4] dependencies ready

if not exist rag\vectorstore\db_faiss (
    echo [3/4] building vectorstore...
    python build_vectorstore.py
    if errorlevel 1 (
        echo [WARN] vectorstore build failed - check legal_pdfs/ for PDF files
    )
) else (
    echo [3/4] vectorstore exists, skip
)

if not exist frontend\dist (
    echo [4/4] building frontend...
    pushd frontend
    call npm install --silent 2>nul
    call npm run build 2>nul
    popd
    if not exist frontend\dist (
        echo [WARN] frontend build may have failed
        echo        cd frontend ^&^& npm install ^&^& npm run build
    )
) else (
    echo [4/4] frontend exists, skip
)

echo.
echo ========================================
echo   Ready! Visit http://localhost:8000
echo ========================================
echo.

start "" http://localhost:8000
python main.py
pause
