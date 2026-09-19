# Frontend

Optional browser-demo boundary. It is outside the mandatory assignment path and communicates with `ai/` only through the AI service API (never imports AI source).

## Chạy

```bash
cd frontend
cp .env.example .env.local   # sửa URL nếu API không ở 127.0.0.1:8000
npm install
npm run dev                  # mở http://localhost:3000
```

Trang duy nhất: nhập bình luận → gọi `POST /predict` → hiện nhãn, confidence,
điểm 3 lớp, cờ `uncertain`, model và `request_id`; kèm trạng thái `/health-check`.
`npm run build` để kiểm tra production build.
