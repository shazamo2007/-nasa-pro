@echo off
chcp 65001 >nul
title Weather App Runner
echo ===============================
echo    تشغيل تطبيق تحليل الطقس
echo ===============================
echo.

echo ✅ فحص تثبيت Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python غير مثبت!
    echo   يرجى تثبيت Python من python.org
    pause
    exit /b 1
)

echo فحص تثبيت المكتبات...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo 📦 تثبيت المكتبات المطلوبة...
    python -m pip install -r requirements.txt

)

echo.
echo  تشغيل التطبيق...
echo  سيفتح المتصفح تلقائياً خلال 5 ثواني...
echo   لإيقاف التطبيق: Ctrl+C
echo.

timeout /t 5 /nobreak >nul

start "" "http://localhost:8000/docs"
start "" "http://localhost:8000"

uvicorn main:app --reload --host 0.0.0.0 --port 8000
