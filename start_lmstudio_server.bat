@echo off
title LM Studio Server - Gemma-3-27B
echo ========================================
echo Starting LM Studio Server (CLI mode)
echo ========================================
echo.

REM Шаг 1: Загрузка модели в память
echo [1/2] Loading model: Gemma-3-27B...
"C:\Users\PC\.lmstudio\bin\lms.exe" load google/gemma-3-27b --gpu max --yes
echo.

REM Шаг 2: Запуск сервера
echo [2/2] Starting server on port 1234...
"C:\Users\PC\.lmstudio\bin\lms.exe" server start --port 1234 --cors --verbose
echo.
echo ========================================
echo Server started: http://localhost:1234
echo Press Ctrl+C to stop
echo ========================================
pause
