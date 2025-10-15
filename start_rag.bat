@echo off
chcp 65001 >nul
cls
echo ═══════════════════════════════════════════════════════════
echo    🧠 SMART RAG Agent 2025 - Qwen3 Function Calling
echo ═══════════════════════════════════════════════════════════
echo.
echo    Умный агент с многоуровневой логикой поиска
echo    Qwen3 сама выбирает инструменты и стратегию
echo.
echo ═══════════════════════════════════════════════════════════
echo.

REM Закрытие предыдущих процессов Python (Gradio)
echo [0/5] Закрытие предыдущих процессов...
taskkill /F /FI "WINDOWTITLE eq Advanced RAG 2025*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq SMART RAG Agent*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq *rag_web_modern.py*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq *rag_smart_qwen.py*" >nul 2>&1
REM Закрыть процессы на портах 7860 и 7861
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7861') do taskkill /F /PID %%a >nul 2>&1
timeout /t 2 /nobreak >nul
echo [OK] Предыдущие процессы закрыты
echo.

REM Проверка, что LM Studio запущен
echo [1/5] Проверка LM Studio...
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
echo [2/5] Проверка Qwen3-30B-A3B...
curl -s http://localhost:1234/v1/models 2>nul | findstr "qwen" >nul
if errorlevel 1 (
    echo [!] Qwen3-30B-A3B не найдена в LM Studio!
    echo [!] Загрузите модель: qwen/qwen3-30b-a3b-2507
    echo.
    echo [*] Пропускаю проверку, продолжаю запуск...
) else (
    echo [OK] Qwen3-30B-A3B загружена
)

echo.
echo [3/5] Проверка Ultimate базы данных...
if exist "chroma_db_ultimate" (
    echo [OK] Ultimate база найдена
) else (
    echo [!] Ultimate база не найдена!
    echo [*] Создайте базу: python create_ultimate_db.py
    echo [*] Или используйте обычный режим: rag_web_modern.py
    echo.
    pause
    exit /b 1
)

echo.
echo [4/5] Запуск SMART RAG Agent...
echo [*] URL: http://localhost:7861
echo [*] Режим: Qwen3 Function Calling
echo [*] База: Ultimate (multilingual-e5-large)
echo [*] Контекст: 16000 токенов
echo.

REM Запуск SMART Agent в новом окне
start "SMART RAG Agent 2025" cmd /k "cd /d "%~dp0" && python rag_smart_qwen.py"

echo [5/5] Ожидание инициализации веб-сервера...
timeout /t 5 /nobreak >nul

REM Открытие браузера
echo [*] Открываю браузер...
start http://localhost:7861

echo.
echo ═══════════════════════════════════════════════════════════
echo    🚀 SMART RAG Agent запущен!
echo ═══════════════════════════════════════════════════════════
echo.
echo [+] SMART Agent: http://localhost:7861
echo [+] LM Studio API: http://localhost:1234
echo [+] Модель: Qwen3-30B-A3B (function calling)
echo [+] База: Ultimate (2.2GB embeddings)
echo.
echo 💡 После запуска нажмите "Запустить SMART Agent"
echo    Ultimate база загрузится автоматически!
echo.
echo ℹ️  Для обычного режима используйте: rag_web_modern.py
echo.
echo Нажмите любую клавишу для выхода...
pause >nul
