@echo off
chcp 65001 > nul
echo ================================================
echo     VeriDoc - Kiem tra dao van and AI
echo ================================================
echo.

python --version > nul 2>&1
if errorlevel 1 (
    echo [LOI] Chua cai Python! Tai tai: https://python.org/downloads
    echo Nho tick "Add Python to PATH" khi cai dat.
    pause
    exit /b
)

echo [1/3] Kiem tra va cai thu vien...
pip install flask flask-cors anthropic mammoth python-docx requests -q 2>nul
echo      Xong!

if not exist "uploads" mkdir uploads

echo [2/3] Khoi dong server...
echo [3/3] Mo trinh duyet...
echo.
echo ================================================
echo  App dang chay tai: http://localhost:5000
echo  Nhan Ctrl+C de dung app
echo ================================================
echo.

start "" /b timeout /t 2 >nul
start "" "http://localhost:5000"

python app.py
pause
