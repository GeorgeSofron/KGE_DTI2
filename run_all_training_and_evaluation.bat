@echo off
REM Master Training and Evaluation Runner (Batch)
REM ================================================

setlocal enabledelayedexpansion

echo.
echo ======================================================================
echo MASTER TRAINING AND EVALUATION RUNNER (Batch)
echo ======================================================================
echo Working directory: %CD%
echo.

REM Run Python script directly
python run_all_training_and_evaluation.py %*

REM Capture exit code
if %ERRORLEVEL% equ 0 (
    echo.
    exit /b 0
) else (
    echo.
    exit /b 1
)
