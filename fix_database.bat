@echo off
chcp 65001 >nul
echo ═══════════════════════════════════════════════════════════
echo    Исправление базы данных
echo ═══════════════════════════════════════════════════════════
echo.

echo [1/3] Остановка Python процессов...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 3 /nobreak >nul

cd /d "%~dp0"

echo [2/3] Удаление старой базы...
if exist "chroma_db_космоэнергетика" (
    rmdir /S /Q "chroma_db_космоэнергетика"
    echo    ✓ Удалена chroma_db_космоэнергетика
)

echo [3/3] Переименование правильной базы...
if exist "chroma_db_test" (
    ren "chroma_db_test" "chroma_db_космоэнергетика"
    echo    ✓ chroma_db_test → chroma_db_космоэнергетика
) else (
    echo    ✗ chroma_db_test не найдена!
)

echo.
echo ═══════════════════════════════════════════════════════════
echo    Готово! Теперь запустите start_rag.bat
echo ═══════════════════════════════════════════════════════════
pause
