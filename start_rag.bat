@echo off
chcp 65001 >nul
cls
echo ═══════════════════════════════════════════════════════════
echo    Advanced RAG 2025 - Запуск системы
echo ═══════════════════════════════════════════════════════════
echo.

REM Закрытие предыдущих процессов Python (Gradio)
echo [0/4] Закрытие предыдущих процессов...
taskkill /F /FI "WINDOWTITLE eq Advanced RAG 2025*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq *rag_web_modern.py*" >nul 2>&1
REM Закрыть все Python процессы использующие порт 7860
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860') do taskkill /F /PID %%a >nul 2>&1
timeout /t 2 /nobreak >nul
echo [OK] Предыдущие процессы закрыты
echo.

REM Проверка, что LM Studio запущен
echo [1/4] Проверка LM Studio...
curl -s http://localhost:1234/v1/models >nul 2>&1
if errorlevel 1 (
    echo.
    echo [!] LM Studio не запущен!
    echo [*] Запускаю LM Studio сервер...
    echo.
    start /B cmd /c start_lmstudio_server.bat
    timeout /t 10 /nobreak >nul
) else (
    echo [OK] LM Studio работает
)

echo.
echo [2/4] Запуск веб-интерфейса RAG...
echo [*] URL: http://localhost:7860
echo.

REM Запуск Python скрипта в новом окне
start "Advanced RAG 2025" cmd /k "cd /d "%~dp0" && python rag_web_modern.py"

echo [3/4] Ожидание инициализации веб-сервера...
timeout /t 5 /nobreak >nul

REM Открытие браузера
echo [4/4] Открываю браузер...
start http://localhost:7860

echo.
echo ═══════════════════════════════════════════════════════════
echo    RAG система запущена!
echo ═══════════════════════════════════════════════════════════
echo.
echo [+] Веб-интерфейс: http://localhost:7860
echo [+] LM Studio API: http://localhost:1234
echo.
echo Нажмите любую клавишу для выхода...
pause >nul
