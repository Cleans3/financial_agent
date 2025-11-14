# Financial Agent - Vietnamese Stock Market Assistant 🇻🇳📈

**Agent AI tư vấn đầu tư chứng khoán Việt Nam** - Hệ thống phân tích thị trường chứng khoán thông minh sử dụng LangGraph, VnStock API và LLM.

---

## ✨ Tính năng

### 📊 Thông tin doanh nghiệp

- ✅ **Thông tin công ty**: Tên công ty, ngành nghề, vốn điều lệ, lịch sử
- ✅ **Cổ đông lớn**: Top cổ đông với tỷ lệ sở hữu chi tiết
- ✅ **Ban lãnh đạo**: Danh sách lãnh đạo và tỷ lệ sở hữu
- ✅ **Công ty con**: Công ty con/liên kết với tỷ lệ nắm giữ
- ✅ **Sự kiện công ty**: Chia cổ tức, ĐHCĐ, tăng vốn...

### 📈 Dữ liệu thị trường

- ✅ **Giá lịch sử (OHLCV)**: Open, High, Low, Close, Volume
  - Theo ngày cụ thể: `start_date` và `end_date`
  - Theo khoảng thời gian: `3M`, `6M`, `1Y`
  - Hiển thị chi tiết dưới dạng bảng

### 📉 Phân tích kỹ thuật

- ✅ **SMA (Simple Moving Average)**: Phân tích xu hướng giá
  - Tính SMA với window tùy chỉnh (SMA-9, SMA-20, SMA-50...)
  - So sánh giá với SMA, xác định xu hướng
  - Hiển thị bảng chi tiết theo từng ngày
- ✅ **RSI (Relative Strength Index)**: Đánh giá quá mua/quá bán
  - RSI > 70: Quá mua (cảnh báo giảm)
  - RSI < 30: Quá bán (cơ hội tăng)
  - Hiển thị bảng chi tiết với trạng thái

### 🎯 Định dạng trả lời

- 📋 **Bảng Markdown** với dữ liệu chi tiết, dễ đọc
- 📊 **Thống kê tổng quan** sau mỗi bảng
- 💡 **Phân tích và kết luận** chuyên nghiệp

---

## 🏗️ Kiến trúc hệ thống

### Tech Stack:

- **Backend**: FastAPI (REST API)
- **Agent Framework**: LangChain + LangGraph (ReAct Pattern)
- **LLM Providers**:
  - ☁️ Google Gemini (Cloud)
  - 🖥️ Ollama (Local)
- **Data Source**: VnStock3 API (Free)
- **Technical Analysis**: TA-Lib
- **Frontend**: React + Vite + TailwindCSS

### Cấu trúc thư mục:

```
financial_agent/
├── src/
│   ├── agent/          # LangGraph Agent
│   │   ├── financial_agent.py
│   │   ├── state.py
│   │   └── prompts/
│   ├── tools/          # 8 Tools
│   │   ├── vnstock_tools.py    # 6 VnStock tools
│   │   └── technical_tools.py  # 2 Technical tools
│   ├── llm/            # LLM Factory
│   │   ├── llm_factory.py
│   │   └── config.py
│   └── api/            # FastAPI
│       └── app.py
├── frontend/           # React UI
│   ├── src/
│   │   ├── components/
│   │   └── App.jsx
│   └── package.json
├── tests/              # Unit Tests
├── test_auto.py        # Automated Test Script
└── requirements.txt
```

---

## 🚀 Cài đặt và Chạy

### Bước 1: Clone và cài đặt Dependencies

```bash
# Clone hoặc cd vào thư mục
cd financial_agent

# Tạo virtual environment
python -m venv venv

# Kích hoạt venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Cấu hình LLM Provider

Bạn có thể chọn 1 trong 2 provider:

#### Option 1: Google Gemini (Recommended) ☁️

**Ưu điểm**: Nhanh, mạnh mẽ, không cần GPU

1. Lấy API key miễn phí tại: https://aistudio.google.com/apikey
2. Cập nhật file `.env`:

```env
# Google Gemini
GOOGLE_API_KEY=your_api_key_here
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.0-flash
```

#### Option 2: Ollama (Local) 🖥️

**Ưu điểm**: Chạy offline, bảo mật, miễn phí hoàn toàn

**Yêu cầu**: RAM >= 8GB (khuyến nghị 16GB), GPU có VRAM >= 4GB (tùy chọn)

**Bước 1: Tải và cài đặt Ollama**

- **Windows**:

  1. Tải tại: https://ollama.com/download/windows
  2. Chạy file `OllamaSetup.exe`
  3. Cài đặt theo hướng dẫn (Next → Next → Install)

- **macOS**:

  ```bash
  brew install ollama
  ```

- **Linux**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

**Bước 2: Khởi động Ollama**

```bash
# Chạy Ollama server (sẽ tự động chạy ở background trên Windows)
ollama serve
```

**Bước 3: Pull model**

Chọn 1 trong các model sau (theo cấu hình máy):

```bash
# Model nhỏ (RAM 4-8GB) - Tốc độ nhanh
ollama pull qwen2.5:3b

# Model trung bình (RAM 8-16GB) - Cân bằng
ollama pull llama3.1:8b
ollama pull qwen2.5:7b

# Model lớn (RAM 16GB+, GPU 8GB+) - Chất lượng cao
ollama pull qwen2.5:14b
ollama pull llama3.1:70b
```

**Bước 4: Kiểm tra model đã cài**

```bash
ollama list
```

**Bước 5: Cập nhật `.env`**

```env
# Ollama Local
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b       # Thay bằng model bạn đã pull
OLLAMA_BASE_URL=http://localhost:11434
```

**Lưu ý Ollama**:

- Model `qwen2.5:3b` (3B parameters) cần ~4GB RAM
- Model `llama3.1:8b` (8B parameters) cần ~8GB RAM
- Nếu gặp lỗi "out of memory", thử model nhỏ hơn hoặc chuyển sang Gemini
- Kiểm tra Ollama đang chạy: `ollama list`

---

## 🎮 Chạy ứng dụng

### Backend API

```bash
# Activate venv (nếu chưa)
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Chạy FastAPI server
uvicorn src.api.app:app --reload

# Server chạy tại: http://localhost:8000
```

### Frontend (React)

```bash
# Terminal mới, cd vào frontend
cd frontend

# Cài đặt dependencies (lần đầu)
npm install

# Chạy dev server
npm run dev

# Frontend chạy tại: http://localhost:5173
```

### Test API bằng curl

```bash
# Test endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Thông tin về VNM\"}"
```

---

## 🧪 Testing

### Test tự động với file Excel

```bash
# Tạo file Excel mẫu với câu hỏi test
python create_sample_test.py

# Chạy test tự động (đảm bảo backend đang chạy)
python test_auto.py test_questions_sample.xlsx

# Kết quả sẽ được lưu trong test_results_[timestamp].xlsx
```

Chi tiết xem file `TESTING.md`

---

## 📚 Sử dụng

### Các câu hỏi mẫu:

**Thông tin công ty:**

```
- "Thông tin về công ty VNM"
- "VCB thuộc ngành gì?"
```

**Cổ đông & Lãnh đạo:**

```
- "Cổ đông lớn của VCB là ai?"
- "Ban lãnh đạo HPG gồm những ai?"
- "VNM có công ty con nào?"
```

**Sự kiện:**

```
- "Sự kiện gần đây của FPT"
- "VCB có chia cổ tức không?"
```

**Dữ liệu giá:**

```
- "Giá VCB 3 tháng gần nhất"
- "OHLCV của HPG từ đầu năm 2024"
```

**Phân tích kỹ thuật:**

```
- "Tính SMA-20 cho HPG"
- "Tính SMA-9 và SMA-20 của TCB từ đầu tháng 11"
- "RSI của VIC hiện tại"
- "HPG có quá mua không?"
```

**Phân tích tổng hợp:**

```
- "Phân tích toàn diện về VNM"
- "So sánh giá VCB và TCB trong 6 tháng"
```

---

## 🔧 Cấu hình nâng cao

### Thay đổi LLM Provider

Chỉnh sửa `.env`:

```env
# Gemini
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your_key_here

# Ollama
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434
```

**Lưu ý**: Phải restart server sau khi thay đổi `.env`

### Ollama Troubleshooting

**Lỗi: "Connection refused"**

```bash
# Kiểm tra Ollama đang chạy
ollama list

# Nếu không chạy, khởi động lại
ollama serve
```

**Lỗi: "Out of memory"**

- Thử model nhỏ hơn: `ollama pull qwen2.5:3b`
- Hoặc chuyển sang Gemini

**Lỗi: "Model not found"**

```bash
# Kiểm tra model đã pull chưa
ollama list

# Pull model
ollama pull qwen2.5:3b
```

### Tùy chỉnh System Prompt

Chỉnh sửa file: `src/agent/prompts/system_prompt.txt`

Restart server để áp dụng thay đổi.

---

## 📡 API Documentation

### Endpoint: `POST /api/chat`

**Request:**

```json
{
  "question": "Thông tin về VNM"
}
```

**Response:**

```json
{
  "answer": "VNM là Công ty Cổ phần Sữa Việt Nam (Vinamilk)...\n\n| Thông tin | Giá trị |\n|-----------|---------|..."
}
```

### Swagger UI

Mở trình duyệt: **http://localhost:8000/docs**

### Example với Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat",
    json={"question": "Giá VCB 3 tháng gần nhất"}
)
print(response.json()["answer"])
```

### Example với cURL

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tính SMA-20 cho HPG"}'
```

---

## 🛠️ Chi tiết 8 Tools

### 1. get_company_info

- **Mô tả**: Thông tin tổng quan về công ty
- **Input**: `ticker` (VNM, VCB, HPG...)
- **Output**: Tên, ngành, vốn điều lệ, lịch sử công ty

### 2. get_shareholders

- **Mô tả**: Danh sách cổ đông lớn
- **Input**: `ticker`
- **Output**: Top 10 cổ đông, tỷ lệ sở hữu, số lượng CP

### 3. get_officers

- **Mô tả**: Ban lãnh đạo công ty
- **Input**: `ticker`
- **Output**: Danh sách lãnh đạo, chức vụ, tỷ lệ sở hữu

### 4. get_subsidiaries

- **Mô tả**: Công ty con và công ty liên kết
- **Input**: `ticker`
- **Output**: Danh sách công ty con, tỷ lệ nắm giữ

### 5. get_company_events

- **Mô tả**: Sự kiện của công ty
- **Input**: `ticker`
- **Output**: 20 sự kiện gần nhất (cổ tức, ĐHCĐ, tăng vốn...)

### 6. get_historical_data

- **Mô tả**: Dữ liệu giá lịch sử (OHLCV)
- **Input**: `ticker`, `start_date`, `end_date` hoặc `period`
- **Output**: Bảng OHLCV chi tiết + thống kê

### 7. calculate_sma

- **Mô tả**: Simple Moving Average
- **Input**: `ticker`, `window` (mặc định 20)
- **Output**: Bảng SMA theo ngày + phân tích xu hướng

### 8. calculate_rsi

- **Mô tả**: Relative Strength Index
- **Input**: `ticker`, `window` (mặc định 14)
- **Output**: Bảng RSI theo ngày + đánh giá quá mua/quá bán

---

## 📊 Response Format

Tools trả về JSON chuẩn:

```json
{
  "success": true,
  "ticker": "VNM",
  "detailed_data": [
    { "date": "2024-11-01", "close": 85.5, "sma_20": 84.2 },
    { "date": "2024-11-04", "close": 86.0, "sma_20": 84.5 }
  ],
  "analysis": {
    "trend": "TĂNG",
    "signal": "positive"
  },
  "message": "Đã tính SMA-20 cho VNM thành công"
}
```

Agent sẽ chuyển đổi JSON này thành bảng Markdown đẹp mắt.

---

## 🎓 Học thêm

### VnStock API

- Documentation: https://vnstocks.com/docs/vnstock
- GitHub: https://github.com/thinh-vu/vnstock

### LangChain & LangGraph

- LangChain Docs: https://python.langchain.com/
- LangGraph Tutorial: https://langchain-ai.github.io/langgraph/

### Technical Analysis

- TA-Lib: https://ta-lib.org/
- Investopedia: https://www.investopedia.com/

### Ollama

- Website: https://ollama.com/
- Model Library: https://ollama.com/library
- GitHub: https://github.com/ollama/ollama

---

## 🐛 Troubleshooting

### Backend không chạy

```bash
# Kiểm tra Python version (cần >= 3.9)
python --version

# Kiểm tra dependencies
pip list | grep langchain

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Frontend không chạy

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### API trả lỗi

```bash
# Kiểm tra logs
# Server sẽ in ra lỗi chi tiết trong terminal

# Test trực tiếp tools
python -c "from src.tools.vnstock_tools import get_company_info; print(get_company_info('VNM'))"
```

### Ollama lỗi

```bash
# Kiểm tra service
ollama list

# Restart service
# Windows: Tìm Ollama trong Task Manager → Restart
# Linux/Mac:
sudo systemctl restart ollama

# Test model
ollama run qwen2.5:3b "Hello"
```

---

## 🎯 Roadmap

- [ ] Thêm tools: Financial Ratios (P/E, ROE, ROA...)
- [ ] Thêm tools: News scraping
- [ ] Thêm charts visualization
- [ ] Deploy lên cloud (Vercel + Railway)
- [ ] Mobile app (React Native)
- [ ] Real-time price updates (WebSocket)

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 👨‍💻 Author

**Financial Agent** - AI Stock Market Assistant for Vietnam

Built with ❤️ using LangGraph, VnStock, and modern AI technologies

**Project**: AI Intern 2025  
**Contact**: [Your contact info]

---

## 🌟 Acknowledgments

- VnStock team for the amazing free API
- LangChain team for the powerful framework
- Ollama team for local LLM support
- Google for Gemini API

---

**Happy Trading! 📈🚀**
