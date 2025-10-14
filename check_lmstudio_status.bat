@echo off
echo ========================================
echo LM Studio Status Check
echo ========================================
echo.

echo Loaded Models:
echo ----------------------------------------
"C:\Users\PC\.lmstudio\bin\lms.exe" ps
echo.

echo Server Status:
echo ----------------------------------------
"C:\Users\PC\.lmstudio\bin\lms.exe" server status
echo.

echo ========================================
pause
