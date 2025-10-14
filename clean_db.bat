@echo off
chcp 65001 >nul
echo ═══════════════════════════════════════════════════════════
echo    Очистка баз данных ChromaDB
echo ═══════════════════════════════════════════════════════════
echo.
echo ВНИМАНИЕ: Это удалит все векторные базы данных!
echo Вам придется заново создавать базу при следующем запуске.
echo.
pause
echo.

echo Закрытие всех Python процессов...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo Удаление баз данных...
for /d %%i in ("%~dp0chroma_db_*") do (
    echo Удаляю: %%~nxi
    rmdir /S /Q "%%i" 2>nul
)

echo.
echo ✅ Готово! Базы данных удалены.
echo.
pause
