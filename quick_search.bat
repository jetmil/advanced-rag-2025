@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ======================================================================
echo БЫСТРЫЙ ПОИСК В ТЕКСТОВОМ ФАЙЛЕ (аналог grep)
echo ======================================================================
echo.

:LOOP
set /p query="Введите запрос (или 'exit' для выхода): "

if /i "%query%"=="exit" goto END
if "%query%"=="" goto LOOP

echo.
echo Ищем: %query%
echo ----------------------------------------------------------------------

findstr /i /n /c:"%query%" cosmic_texts.txt

if errorlevel 1 (
    echo ❌ Ничего не найдено
) else (
    echo ----------------------------------------------------------------------
    echo ✅ Найдено совпадения выше
)

echo.
goto LOOP

:END
echo Пока!
pause
