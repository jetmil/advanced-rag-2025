@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ======================================================================
echo 🧠 SMART RAG Agent 2025 - Qwen3 Function Calling
echo ======================================================================
echo.
echo ✨ Умный агент с многоуровневым поиском
echo 🗄️ Автозагрузка Ultimate базы (multilingual-e5-large)
echo 🤖 Qwen3-30B-A3B сам выбирает инструменты
echo.
echo ⚠️  ВАЖНО: Запустите LM Studio и загрузите Qwen3-30B-A3B!
echo.
pause

echo.
echo 🚀 Запуск SMART Agent...
echo.

REM Убиваем старые процессы (если были)
taskkill /F /IM python.exe /FI "WINDOWTITLE eq SMART*" 2>nul

REM Запускаем агент
python rag_smart_qwen.py

REM Если Python завершился с ошибкой
if errorlevel 1 (
    echo.
    echo ❌ Ошибка запуска! Проверьте:
    echo    1. LM Studio запущен?
    echo    2. Qwen3-30B-A3B загружена?
    echo    3. Ultimate база создана? ^(create_ultimate_db.py^)
    echo.
    pause
)
