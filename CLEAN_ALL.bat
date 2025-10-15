@echo off
chcp 65001 >nul
echo ═══════════════════════════════════════════════════════════
echo    ПОЛНОЕ УДАЛЕНИЕ ВСЕХ БАЗ ДАННЫХ
echo ═══════════════════════════════════════════════════════════
echo.

echo [1/2] Остановка всех Python процессов...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 3 /nobreak >nul

cd /d "%~dp0"

echo [2/2] Удаление ВСЕХ баз данных...
for /d %%i in (chroma_db*) do (
    echo    Удаляю: %%i
    rmdir /S /Q "%%i"
)

echo.
echo ✅ ВСЕ базы удалены!
echo.
echo Теперь запустите start_rag.bat и создайте НОВУЮ базу
echo через кнопку "✨ Инициализировать новую"
echo.
pause
