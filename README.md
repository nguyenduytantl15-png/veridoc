# VeriDoc — Phát hiện đạo văn & AI

Web app phân tích tài liệu Word để phát hiện nội dung đạo văn và nội dung do AI tạo ra.

## Tính năng

- Upload file Word (.docx, .doc, .txt)
- Loại trừ thông minh nội dung từ file mẫu
- Phát hiện AI với Claude API (~90% chính xác) hoặc heuristic cục bộ (~70%)
- Quét đạo văn tự động với SerpAPI + link Google
- Hiển thị % và trích dẫn nguồn từng đoạn

---

## Cách chạy trên máy tính (Local)

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Chạy app
python app.py

# 3. Mở trình duyệt
# http://localhost:5000
```

---

## Deploy lên Render (Miễn phí — Share link cho mọi người dùng)

### Bước 1: Tạo tài khoản GitHub
Nếu chưa có, đăng ký tại https://github.com

### Bước 2: Upload code lên GitHub
1. Tạo repository mới tại https://github.com/new (đặt tên: `veridoc`)
2. Tải GitHub Desktop tại https://desktop.github.com
3. Clone repo → Copy tất cả file vào thư mục → Commit & Push

**Hoặc dùng lệnh git:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/TEN_BAN/veridoc.git
git push -u origin main
```

### Bước 3: Deploy lên Render
1. Vào https://render.com → Đăng ký miễn phí
2. Click **"New +"** → **"Web Service"**
3. Kết nối tài khoản GitHub
4. Chọn repo `veridoc`
5. Cấu hình:
   - **Name:** veridoc (hoặc tên bạn muốn)
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
   - **Plan:** Free
6. Click **"Create Web Service"**
7. Đợi 3-5 phút build xong → Render sẽ cấp link dạng `https://veridoc-xxxx.onrender.com`

**Chia sẻ link đó cho mọi người là xong!** 🎉

### Lưu ý về Render Free Plan
- App sẽ "ngủ" sau 15 phút không dùng → lần đầu load mất ~30 giây
- Để app luôn online, dùng UptimeRobot (miễn phí) ping mỗi 14 phút

---

## Lấy API Keys (Tùy chọn)

### Anthropic API Key (phát hiện AI chính xác hơn)
1. Vào https://console.anthropic.com
2. Đăng ký → vào **API Keys** → **Create Key**
3. Copy key dạng `sk-ant-api03-...`
4. Dán vào ô "Anthropic API Key" trong app

### SerpAPI Key (quét đạo văn tự động)
1. Vào https://serpapi.com
2. Đăng ký → 100 lượt tìm kiếm/tháng miễn phí
3. Vào dashboard → copy API key
4. Dán vào ô "SerpAPI Key" trong app

---

## Cấu trúc dự án

```
veridoc/
├── app.py              # Backend Flask chính
├── requirements.txt    # Python dependencies
├── Procfile           # Lệnh chạy cho Render/Heroku
├── render.yaml        # Cấu hình Render
├── templates/
│   └── index.html     # Giao diện web
└── uploads/           # Thư mục tạm (tự tạo)
```

---

## Deploy lên các nền tảng khác

### Railway (miễn phí $5/tháng credit)
1. Vào https://railway.app
2. New Project → Deploy from GitHub repo
3. Tự detect Python và deploy

### Heroku
```bash
heroku login
heroku create veridoc-app
git push heroku main
```

### VPS (Timedivine, DigitalOcean...)
```bash
# Cài nginx + gunicorn
sudo apt install nginx python3-pip
pip install -r requirements.txt
gunicorn app:app --daemon --workers 2 --bind 127.0.0.1:5000

# Cấu hình nginx reverse proxy
# Trỏ domain vào server
```

---

## Giới hạn và lưu ý

- Kết quả phân tích mang tính tham khảo, không phải kết luận cuối cùng
- Nên kết hợp với đánh giá thủ công
- File tối đa 16MB
- AI detection không hoàn hảo 100%
