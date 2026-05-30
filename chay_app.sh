#!/bin/bash
echo "================================================"
echo "    VeriDoc - Kiểm tra đạo văn & AI"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[LỖI] Chưa cài Python3! Tải tại: https://python.org/downloads"
    exit 1
fi

echo "[1/3] Cài thư viện..."
pip3 install flask flask-cors anthropic mammoth python-docx requests -q 2>/dev/null || \
pip3 install flask flask-cors anthropic mammoth python-docx requests -q --break-system-packages 2>/dev/null
echo "      Xong!"

mkdir -p uploads

echo "[2/3] Khởi động server..."
echo "[3/3] Mở trình duyệt..."
echo ""
echo "================================================"
echo " App đang chạy tại: http://localhost:5000"
echo " Nhấn Ctrl+C để dừng"
echo "================================================"
echo ""

# Open browser
sleep 1.5
if command -v open &> /dev/null; then
    open "http://localhost:5000"      # macOS
elif command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:5000"  # Linux
fi

python3 app.py
