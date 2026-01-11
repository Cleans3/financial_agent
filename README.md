# Financial Agent - Vietnamese Stock Market Assistant 🇻🇳📈

**Agent AI tư vấn đầu tư chứng khoán Việt Nam** - Hệ thống phân tích thị trường chứng khoán thông minh sử dụng LangGraph, VnStock API và LLM.

---

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Database Setup](#database-setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Features](#features)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## ⚡ Quick Start

### Minimum Setup (5 minutes)

```bash
# 1. Clone and navigate to project
git clone <repo-url>
cd financial_agent_fork

# 2. Create Python virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template
cp .env.example .env
# Edit .env with your settings

# 5. Setup database (PostgreSQL required)
# See Database Setup section below

# 6. Run the API server
python main.py
```

Visit `http://localhost:8000/docs` to test the API.

---

## 🖥️ System Requirements

### Minimum Requirements

- **Python**: 3.9 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 5GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux

### External Services Required

1. **PostgreSQL Database** (v12 or higher)
   - Local installation or cloud service (AWS RDS, Azure Database, etc.)
   - At least 2GB storage recommended

2. **LLM Provider** (choose one)
   - **Google Gemini**: Free API key from [Google AI Studio](https://aistudio.google.com/apikey)
   - **Ollama**: Local LLM server (free, no API key needed)

3. **Qdrant Vector Database** (choose one)
   - **Qdrant Cloud**: Free tier available at [cloud.qdrant.io](https://cloud.qdrant.io)
   - **Qdrant Local**: Docker container or local installation

4. **Optional: Tesseract OCR**
   - Required only for processing scanned PDF documents
   - [Installation Guide](https://github.com/UB-Mannheim/tesseract/wiki)

---

## 📦 Installation Guide

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd financial_agent_fork
```

### Step 2: Python Environment Setup

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

Verify Python version:
```bash
python --version  # Should be 3.9 or higher
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Installation may take 5-10 minutes due to native dependencies**

#### Optional: Install Tesseract OCR

For processing scanned PDFs and images:

**Windows:**
```bash
# Download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki/Downloads
# Then run setup and add to your .env:
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

### Step 4: Verify Installation

```bash
python -c "import langchain; print('✓ LangChain installed')"
python -c "import fastapi; print('✓ FastAPI installed')"
python -c "import vnstock; print('✓ VnStock installed')"
python -c "import qdrant_client; print('✓ Qdrant client installed')"
```

---

## 🗄️ Database Setup

This project uses **PostgreSQL** as the primary relational database, with **Qdrant** as the vector database for RAG features.

### PostgreSQL Setup

#### Option 1: Local Installation (Recommended for Development)

**Windows:**

1. Download PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
2. Run the installer and follow the installation wizard
3. Remember the superuser password
4. Verify installation:
   ```bash
   psql --version
   ```

5. Connect to PostgreSQL:
   ```bash
   psql -U postgres
   ```

**macOS:**

```bash
# Using Homebrew
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Connect to PostgreSQL
psql postgres
```

**Linux (Ubuntu/Debian):**

```bash
# Update package list
sudo apt-get update

# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Connect to PostgreSQL
sudo -u postgres psql
```

#### Option 2: Docker Container (Recommended for Production)

```bash
# Run PostgreSQL container
docker run --name financial-db \
  -e POSTGRES_USER=financial_user \
  -e POSTGRES_PASSWORD=financial_password \
  -e POSTGRES_DB=financial_agent \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  -d postgres:15

# Verify container is running
docker ps
```

#### Create Database and User

```bash
# Connect to PostgreSQL
psql -U postgres

# Inside psql shell:
CREATE USER financial_user WITH PASSWORD 'financial_password';
CREATE DATABASE financial_agent OWNER financial_user;

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE financial_agent TO financial_user;

# Connect to the new database
\c financial_agent

# Verify connection
\dt
```

**Connection String:**
```
postgresql://financial_user:financial_password@localhost:5432/financial_agent
```

#### Option 3: Cloud PostgreSQL

**AWS RDS:**
1. Go to [AWS RDS Console](https://console.aws.amazon.com/rds/)
2. Click "Create Database"
3. Select PostgreSQL engine
4. Configure settings and note the endpoint
5. Add connection string to `.env`:
   ```
   DATABASE_URL=postgresql://username:password@endpoint:5432/financial_agent
   ```

**Azure Database for PostgreSQL:**
1. Go to [Azure Portal](https://portal.azure.com/)
2. Create new "Azure Database for PostgreSQL"
3. Configure and get connection details
4. Add to `.env`

**Supabase (PostgreSQL as a Service):**
1. Sign up at [supabase.com](https://supabase.com/)
2. Create new project
3. Copy connection string from project settings
4. Add to `.env`:
   ```
   DATABASE_URL=postgresql://[user]:[password]@[host]:[port]/[database]
   ```

### Database Initialization

After PostgreSQL is ready, initialize the application database:

```bash
# Navigate to project root
cd financial_agent_fork

# Run migrations using Alembic
alembic upgrade head
```

**Expected Output:**
```
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade -> xxxxx, Initial migration
```

### Verify Database Setup

```bash
# Connect to database
psql -U financial_user -d financial_agent -h localhost

# List all tables
\dt

# Expected tables:
# - users
# - chat_sessions
# - chat_messages
# - audit_logs
# - document_uploads

# Exit psql
\q
```

---

### Qdrant Vector Database Setup

Qdrant stores vector embeddings for RAG (Retrieval Augmented Generation) features.

#### Option 1: Qdrant Cloud (Recommended for Production)

1. **Sign Up**: Go to [cloud.qdrant.io](https://cloud.qdrant.io/)
2. **Create Cluster**:
   - Click "Create Cluster"
   - Select region (choose closest to your location)
   - Name: `financial-agent` or similar
   - Free tier available for testing

3. **Get Credentials**:
   - Copy the API Key and Cluster URL
   - Add to `.env`:
     ```
     QDRANT_MODE=cloud
     QDRANT_CLOUD_URL=https://your-cluster.qdrant.io
     QDRANT_CLOUD_API_KEY=your-api-key
     ```

4. **Verify Connection**:
   ```bash
   python -c "from qdrant_client import QdrantClient; c = QdrantClient(url='YOUR_URL', api_key='YOUR_KEY'); print('✓ Qdrant connected')"
   ```

#### Option 2: Docker Container (Development)

```bash
# Run Qdrant container
docker run --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  -d qdrant/qdrant

# Verify container
docker ps

# Check web interface
# Visit http://localhost:6333/dashboard
```

**Add to `.env`:**
```
QDRANT_MODE=local
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

#### Option 3: Local Installation (Development)

```bash
# Download and run Qdrant locally
# Visit https://qdrant.tech/documentation/quick-start/ for platform-specific instructions

# macOS:
brew install qdrant

# Linux:
docker run -p 6333:6333 qdrant/qdrant
```

---

## 🔧 Environment Configuration

### Create .env File

```bash
# Copy the template
cp .env.example .env
```

### Complete Configuration

Edit `.env` with all required values:

```dotenv
# ==========================================
# DATABASE CONFIGURATION
# ==========================================
DATABASE_URL=postgresql://financial_user:financial_password@localhost:5432/financial_agent
JWT_SECRET_KEY=your-super-secret-key-change-this-in-production
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password_here

# ==========================================
# LLM PROVIDER CONFIGURATION
# ==========================================
LLM_PROVIDER=gemini          # Options: 'gemini' or 'ollama'
GOOGLE_API_KEY=your_api_key  # Required if using Gemini
LLM_MODEL=gemini-2.5-flash   # Google Gemini model

# OR for Ollama:
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

# LLM Settings
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2048

# ==========================================
# QDRANT VECTOR DATABASE
# ==========================================
QDRANT_MODE=cloud              # 'cloud' or 'local'

# Cloud Settings:
QDRANT_CLOUD_URL=https://your-instance.qdrant.io
QDRANT_CLOUD_API_KEY=your-qdrant-api-key

# OR Local Settings:
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=

# Timeout settings
QDRANT_TIMEOUT_SECONDS=120
QDRANT_RETRY_ATTEMPTS=3
QDRANT_RETRY_DELAY_SECONDS=2.0

# ==========================================
# EMBEDDING CONFIGURATION
# ==========================================
EMBEDDING_MODEL_FINANCIAL=fin-e5-small
EMBEDDING_MODEL_GENERAL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=50

# ==========================================
# RAG CONFIGURATION
# ==========================================
ENABLE_RAG=True
RAG_PRIORITY_MODE=personal-first
RAG_SIMILARITY_THRESHOLD=0.1
RAG_TOP_K_RESULTS=20
RAG_MIN_RELEVANCE=0.3
RAG_MAX_DOCUMENTS=5

# ==========================================
# FEATURE FLAGS
# ==========================================
DEBUG=False
ENABLE_TOOLS=True
ENABLE_SUMMARIZATION=True
ENABLE_QUERY_REWRITING=True

# ==========================================
# API CONFIGURATION
# ==========================================
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://localhost:8000

# ==========================================
# RATE LIMITING
# ==========================================
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD_MINUTES=60
```

### Validate Configuration

```bash
python -c "from src.core.config import settings; print('✓ Configuration loaded'); print(f'DB: {settings.DATABASE_URL}'); print(f'LLM: {settings.LLM_PROVIDER}')"
```

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

### 📄 Xử lý Tài liệu

- ✅ **Phân tích Báo cáo Tài chính (Hình ảnh)**: 
  - OCR từ ảnh PDF/PNG/JPG
  - Phân loại báo cáo: BCDN, KQKD, Dòng tiền, Chỉ số
  - Trích xuất dữ liệu + tạo bảng Markdown
  - Phân tích Gemini AI chi tiết

- ✅ **Xử lý File PDF**:
  - Trích xuất text từ PDF native
  - OCR tự động cho PDF scanned
  - Bảng và dữ liệu có cấu trúc
  - Phân tích thông minh với Gemini

- ✅ **Phân tích File Excel**:
  - Chuyển đổi thành bảng Markdown
  - Hỗ trợ nhiều sheet
  - Định dạng số chuẩn Việt Nam
  - Phân tích tài chính chi tiết

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
  - ☁️ Google Gemini (Cloud) - cho phân tích tài chính & OCR
  - 🖥️ Ollama (Local) - cho chat & phân tích
- **Data Source**: VnStock3 API (Free)
- **Technical Analysis**: TA-Lib
- **Document Processing**: 
  - pytesseract + OpenCV (OCR)
  - pdfplumber (PDF text extraction)
  - pdf2image (PDF to image conversion)
- **Excel Processing**: openpyxl + pandas
- **Frontend**: React + Vite + TailwindCSS

### Cấu trúc thư mục:

```
financial_agent/
├── src/
│   ├── agent/          # LangGraph Agent
│   │   ├── financial_agent.py
│   │   ├── state.py
│   │   └── prompts/
│   │       ├── system_prompt.txt
│   │       ├── financial_report_prompt.txt
│   │       └── excel_analysis_prompt.txt
│   ├── tools/          # 11+ Tools
│   │   ├── vnstock_tools.py        # 5 VnStock tools
│   │   ├── technical_tools.py      # 2 Technical analysis tools
│   │   ├── financial_report_tools.py  # Financial report analysis (OCR + Gemini)
│   │   ├── pdf_tools.py            # PDF document processing
│   │   └── excel_tools.py          # Excel analysis tools
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
LLM_MODEL=gemini-2.5-flash
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

### Bước 6: Cấu hình Tesseract OCR (cho phân tích báo cáo tài chính)

Tesseract được dùng để OCR hình ảnh báo cáo tài chính. Có thể bỏ qua nếu chỉ dùng Gemini Vision hoặc PDF native.

#### Windows:

1. Tải installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Chạy `tesseract-ocr-w64-setup-v5.x.exe`
3. Cài đặt theo hướng dẫn (mặc định: `C:\Program Files\Tesseract-OCR`)
4. Cập nhật `.env`:

```env
# Optional: Chỉ cần nếu install ở vị trí custom
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

#### Linux (Ubuntu/Debian):

```bash
sudo apt-get install tesseract-ocr libtesseract-dev
```

#### macOS:

```bash
brew install tesseract
```

#### Kiểm tra cài đặt:

```bash
tesseract --version
```

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

### Tải lên và phân tích tài liệu:

**Báo cáo tài chính (Hình ảnh):**

Gửi hình ảnh báo cáo tài chính (BCDN, KQKD, Dòng tiền):
```
- Upload file PNG/JPG của báo cáo
- Agent sẽ OCR + phân tích + tạo bảng Markdown
```

**File PDF:**

Gửi file PDF báo cáo tài chính:
```
- Upload file PDF (native text hoặc scanned)
- Agent sẽ trích xuất text + bảng
- Phân tích chi tiết với AI
```

**File Excel:**

Gửi file Excel dữ liệu tài chính:
```
- Upload file .xlsx/.xls
- Agent sẽ chuyển đổi thành Markdown
- Phân tích dữ liệu tài chính
```

---

## 🚀 Running the Application

### 1. Start Backend API Server

```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Start the FastAPI server
python main.py
```

**Expected Output:**
```
╔══════════════════════════════════════════════════════════════╗
║              Financial Agent API                             ║
║       Vietnamese Stock Market Investment Assistant           ║
╚══════════════════════════════════════════════════════════════╝

🚀 Starting server...
📍 API Server: http://0.0.0.0:8000
📚 API Documentation (Swagger UI): http://0.0.0.0:8000/docs
...
Press CTRL+C to quit
```

### 2. Test API Server

In a new terminal:

```bash
# Test health check
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Thông tin về VNM"}'
```

### 3. Access Swagger UI

Open browser and visit: **http://localhost:8000/docs**

You can test all API endpoints interactively here.

### 4. (Optional) Start Frontend

```bash
# In a new terminal
cd frontend

# Install dependencies if not already done
npm install

# Start development server
npm run dev
```

Frontend will be available at: **http://localhost:5173**

### 5. (Optional) Start Desktop App

```bash
# In a new terminal
cd desktop_app

# Setup (only first time)
npm install

# Start Electron app
npm start
```

---

## 🔧 Advanced Configuration

---

## 🔧 Advanced Configuration

### Switching LLM Providers

Edit `.env` to change which LLM is used:

```env
# Google Gemini (Cloud)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your_api_key_here

# Ollama (Local)
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434
```

**Important**: Restart the server after changing `.env`

### Setting Up Ollama (Local LLM)

```bash
# Download and install from https://ollama.com/

# Start Ollama server
ollama serve

# In another terminal, pull a model
ollama pull qwen2.5:7b

# Verify installation
ollama list
```

### Troubleshooting Ollama

**Error: "Connection refused"**

```bash
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve
```

**Error: "Out of memory"**

- Use a smaller model: `ollama pull qwen2.5:3b`
- Switch to Gemini (cloud-based)

**Error: "Model not found"**

```bash
# List available models
ollama list

# Pull a new model
ollama pull qwen2.5:7b
```

**Recommended Models for Financial Analysis:**

- `qwen2.5:7b` - Best balance of quality and speed
- `llama2:13b` - High quality but slower
- `qwen2.5:3b` - Fast but lower quality

### Getting Google Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and add to `.env`:
   ```
   GOOGLE_API_KEY=your_key_here
   LLM_PROVIDER=gemini
   ```

### Custom System Prompts

Edit these files to customize agent behavior:

- `src/agent/prompts/system_prompt.txt` - Main agent prompt
- `src/agent/prompts/financial_report_prompt.txt` - Financial report analysis
- `src/agent/prompts/excel_analysis_prompt.txt` - Excel data analysis

Restart server to apply changes.

### Fine-tuning LLM Parameters

```env
# Temperature (0.0-1.0): Higher = more creative, Lower = more focused
LLM_TEMPERATURE=0.3

# Maximum length of response
LLM_MAX_TOKENS=2048

# RAG Threshold (0.0-1.0): How relevant documents must be
RAG_SIMILARITY_THRESHOLD=0.1

# Number of documents to retrieve
RAG_TOP_K_RESULTS=20
```

### Installing Additional Tools

#### Install Tesseract OCR (Optional)

Only needed for processing scanned PDFs:

**Windows:**

```
# Download from: https://github.com/UB-Mannheim/tesseract/wiki/Downloads
# Run installer

# Add to .env:
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**macOS:**

```bash
brew install tesseract
```

**Linux:**

```bash
sudo apt-get install tesseract-ocr
```

#### Using TA-Lib for Advanced Technical Analysis

```bash
# Already installed via requirements.txt
# Verify installation
python -c "import talib; print('✓ TA-Lib installed')"
```

---

## 📡 API Endpoints

### Health Check Endpoint

**GET** `/health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-11T10:30:00Z"
}
```

### Chat Endpoint

**POST** `/api/chat`

Ask the financial agent any question about Vietnamese stocks.

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the latest price of VNM stock?"
  }'
```

**Request Body:**
```json
{
  "question": "Your question here",
  "use_rag": true,  // Optional: use RAG for document analysis
  "session_id": "optional_session_id"
}
```

**Response:**
```json
{
  "answer": "VNM (Vinamilk) stock information...\n\n| Date | Close | Volume |\n...",
  "sources": ["VnStock API", "Company data"],
  "processing_time_seconds": 2.5
}
```

### Upload Financial Report

**POST** `/api/upload/financial-report`

Analyze financial reports from images (PNG, JPG, PDF).

```bash
curl -X POST "http://localhost:8000/api/upload/financial-report" \
  -F "file=@financial_report.jpg"
```

**Response:**
```json
{
  "success": true,
  "report_type": "Balance Sheet",
  "company": "ABC Corporation",
  "period": "Q3/2024",
  "extracted_text": "...",
  "markdown_table": "| Item | Value |\n...",
  "analysis": "Financial analysis from AI..."
}
```

### Upload PDF Document

**POST** `/api/upload/pdf`

Analyze PDF financial documents.

```bash
curl -X POST "http://localhost:8000/api/upload/pdf" \
  -F "file=@report.pdf"
```

**Response:**
```json
{
  "success": true,
  "file_name": "report.pdf",
  "total_pages": 5,
  "extracted_text": "...",
  "tables_markdown": "| Table | Data |\n...",
  "analysis": "Detailed financial analysis...",
  "processing_method": "native"
}
```

### Upload Excel File

**POST** `/api/upload/excel`

Analyze Excel financial data files.

```bash
curl -X POST "http://localhost:8000/api/upload/excel" \
  -F "file=@financial_data.xlsx"
```

**Response:**
```json
{
  "success": true,
  "file_name": "financial_data.xlsx",
  "sheet_count": 3,
  "markdown": "# Financial Data Analysis\n\n## Sheet 1: Revenue\n| Month | Amount |\n...",
  "message": "Excel file analysis successful"
}
```

### Interactive API Documentation

Visit **http://localhost:8000/docs** (Swagger UI) to:
- View all available endpoints
- Test endpoints with example data
- See response schemas
- Download API specification

---

## 🐛 Troubleshooting Guide

### Installation Issues

**Error: "Python version too old"**

```bash
# Check your Python version
python --version

# Should be 3.9 or higher. If not, download from python.org
```

**Error: "pip install failed"**

```bash
# Clear pip cache
pip cache purge

# Upgrade pip
python -m pip install --upgrade pip

# Try installing again
pip install -r requirements.txt
```

**Error: "ModuleNotFoundError: No module named 'xxx'"**

```bash
# Reinstall with force-reinstall
pip install -r requirements.txt --force-reinstall

# Or reinstall specific package
pip install langchain --upgrade
```

### Database Connection Issues

**Error: "Connection refused" for PostgreSQL**

```bash
# Check if PostgreSQL is running
# Windows: Services app → PostgreSQL → Should show "Running"
# macOS: brew services list | grep postgres
# Linux: sudo systemctl status postgresql

# If not running, start it:
# Windows: Services app → PostgreSQL → Start
# macOS: brew services start postgresql@15
# Linux: sudo systemctl start postgresql
```

**Error: "database does not exist"**

```bash
# Recreate the database
psql -U postgres
CREATE DATABASE financial_agent;
GRANT ALL PRIVILEGES ON DATABASE financial_agent TO financial_user;
\q

# Run migrations
alembic upgrade head
```

**Error: "Database URL is empty"**

```bash
# Check .env file has DATABASE_URL
cat .env | grep DATABASE_URL

# Should see something like:
# DATABASE_URL=postgresql://financial_user:financial_password@localhost:5432/financial_agent
```

### LLM Provider Issues

**Error: "GOOGLE_API_KEY not configured"**

```bash
# 1. Get API key from: https://aistudio.google.com/apikey
# 2. Add to .env:
GOOGLE_API_KEY=your_actual_key_here
LLM_PROVIDER=gemini

# 3. Restart server
```

**Error: "Ollama connection failed"**

```bash
# Check if Ollama is running
ollama list

# Start Ollama if not running
ollama serve

# Update .env to point to correct URL
OLLAMA_BASE_URL=http://localhost:11434
LLM_PROVIDER=ollama
```

**Error: "Model not found"**

```bash
# List available models
ollama list

# Pull a model
ollama pull qwen2.5:7b

# Set model in .env
OLLAMA_MODEL=qwen2.5:7b
```

### API Server Issues

**Error: "Port 8000 already in use"**

```bash
# Use different port
API_PORT=8001 python main.py

# Or find process using port 8000 and kill it
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -i :8000
kill -9 <PID>
```

**Error: "CORS error" from frontend**

```bash
# Update .env with correct origins
CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://localhost:8000

# Restart server
```

**Error: "No module named 'src'"**

```bash
# Make sure running from project root directory
cd financial_agent_fork

# Verify directory structure
ls -la src/  # Should show src/ folder exists

# Run server from root
python main.py
```

### Qdrant Vector Database Issues

**Error: "Qdrant connection failed"**

```bash
# Check Qdrant is running
curl http://localhost:6333/health

# If not running, start with Docker
docker run --name qdrant -p 6333:6333 qdrant/qdrant

# Or for Qdrant Cloud, update .env
QDRANT_MODE=cloud
QDRANT_CLOUD_URL=https://your-instance.qdrant.io
QDRANT_CLOUD_API_KEY=your-api-key
```

**Error: "Collection not found"**

This is normal for first run. Collections are created automatically when first document is uploaded.

**Error: "Timeout connecting to Qdrant"**

```bash
# Increase timeout in .env
QDRANT_TIMEOUT_SECONDS=300
QDRANT_RETRY_ATTEMPTS=5

# Restart server
```

### File Upload Issues

**Error: "File size too large"**

- Default limit: 50MB per file
- For larger files, split into multiple smaller files
- Or adjust FastAPI settings

**Error: "Unsupported file type"**

- Financial Reports: PNG, JPG, PDF
- Data Files: XLSX, XLS
- PDF: PDF only

### Document Processing Issues

**Error: "OCR failed" or "Tesseract not found"**

```bash
# Option 1: Install Tesseract (see Installation Guide above)
# Option 2: Use Google Gemini Vision API instead (recommended)
# Set in .env:
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_key_here
```

**Error: "PDF extraction failed"**

- Try with a different PDF file
- Ensure PDF is not password-protected
- Scanned PDFs may need OCR (slower)

**Error: "Excel file cannot be read"**

- Verify file is not corrupted
- Save file in .xlsx format (not .xls)
- Check file has proper Excel structure
- Remove unusual blank rows/columns

### Performance Issues

**API is slow to respond**

```bash
# 1. Check if it's LLM latency
# - Switching to faster model (qwen2.5:3b)
# - Or use Gemini instead

# 2. Check if it's database query
# - Add database indexes
# - Check database server is running properly

# 3. Check RAM usage
# - Monitor memory with: Task Manager (Windows), Activity Monitor (macOS), htop (Linux)
# - If low on RAM, reduce model size

# 4. Enable debug mode to see timings
DEBUG=True
```

**High memory usage**

```bash
# Use smaller LLM model
OLLAMA_MODEL=qwen2.5:3b  # Instead of qwen2.5:7b

# Or switch to API-based (cloud) providers
LLM_PROVIDER=gemini
```

### Getting Help

**Check logs for detailed error messages:**

```bash
# Windows: Logs are printed in terminal
# Look for error messages starting with [ERROR]

# Enable verbose logging
DEBUG=True
```

**Test individual components:**

```bash
# Test VnStock API
python -c "from vnstock3 import Vnstock; v = Vnstock(); print(v.listing_companies())"

# Test PostgreSQL
python -c "from src.database.database import SessionLocal; db = SessionLocal(); print('✓ Database connected')"

# Test Qdrant
python -c "from qdrant_client import QdrantClient; c = QdrantClient(':memory:'); print('✓ Qdrant OK')"

# Test LLM
python -c "from src.llm.llm_factory import LLMFactory; llm = LLMFactory.get_llm(); print(llm.invoke('Hello'))"
```

---

## 🛠️ Available Tools Reference

The financial agent has access to these tools for stock market analysis:

### Stock Information Tools

#### 1. get_company_info
Get company overview and profile information
- **Input**: `ticker` (e.g., VNM, VCB, HPG)
- **Output**: Company name, industry, charter capital, history

#### 2. get_shareholders
Retrieve major shareholders information
- **Input**: `ticker`
- **Output**: Top 10 shareholders with ownership percentages

#### 3. get_officers
Get company leadership and management team
- **Input**: `ticker`
- **Output**: Executives, positions, shareholding percentage

#### 4. get_subsidiaries
Find subsidiary and affiliated companies
- **Input**: `ticker`
- **Output**: List of subsidiaries with ownership percentage

#### 5. get_company_events
Get company events and announcements
- **Input**: `ticker`
- **Output**: Recent corporate events (dividends, AGM, capital increases)

### Market Data Tools

#### 6. get_historical_data
Retrieve historical price data (OHLCV)
- **Input**: `ticker`, `start_date`, `end_date` or `period` (3M, 6M, 1Y)
- **Output**: Detailed OHLCV table with statistics
- **Example**: `get_historical_data("VNM", period="3M")`

### Technical Analysis Tools

#### 7. calculate_sma
Calculate Simple Moving Average
- **Input**: `ticker`, `window` (default: 20)
- **Output**: SMA values with trend analysis
- **Example**: `calculate_sma("VNM", window=20)`

#### 8. calculate_rsi
Calculate Relative Strength Index
- **Input**: `ticker`, `window` (default: 14)
- **Output**: RSI values with overbought/oversold signals
- **Example**: `calculate_rsi("HPG", window=14)`

### How to Use Tools in Chat

Simply ask the agent questions, and it will automatically use the appropriate tools:

```
Q: "What is the latest price of VNM?"
→ Uses get_historical_data

Q: "Who are the major shareholders of VCB?"
→ Uses get_shareholders

Q: "Calculate SMA-20 for HPG"
→ Uses calculate_sma with window=20

Q: "Is FPT stock overbought right now?"
→ Uses calculate_rsi to check signal
```

---

## 🏗️ Architecture

### Tech Stack

- **Backend**: FastAPI (REST API)
- **Agent Framework**: LangChain + LangGraph (ReAct Pattern)
- **LLM Providers**:
  - ☁️ Google Gemini (Cloud) - AI analysis & OCR
  - 🖥️ Ollama (Local) - for chat & analysis
- **Data Source**: VnStock3 API (Free)
- **Vector Database**: Qdrant (RAG)
- **Relational Database**: PostgreSQL
- **Technical Analysis**: TA-Lib
- **Document Processing**:
  - pytesseract + OpenCV (OCR for scanned documents)
  - pdfplumber (PDF text extraction)
  - pdf2image (PDF to image conversion)
- **Excel Processing**: openpyxl + pandas
- **Frontend**: React + Vite + TailwindCSS
- **Desktop App**: Electron

### Project Structure

```
financial_agent/
├── src/
│   ├── agent/              # LangGraph Agent
│   │   ├── financial_agent.py
│   │   ├── state.py
│   │   └── prompts/
│   │       ├── system_prompt.txt
│   │       ├── financial_report_prompt.txt
│   │       └── excel_analysis_prompt.txt
│   ├── tools/              # 8+ Analysis Tools
│   │   ├── vnstock_tools.py         # Company & stock data
│   │   ├── technical_tools.py       # SMA, RSI indicators
│   │   ├── financial_report_tools.py # OCR + Gemini analysis
│   │   ├── pdf_tools.py             # PDF processing
│   │   └── excel_tools.py           # Excel analysis
│   ├── llm/                # LLM Factory
│   │   ├── llm_factory.py
│   │   └── llm_config.py
│   ├── database/           # Database Models
│   │   ├── database.py
│   │   └── models.py
│   ├── api/                # REST API Endpoints
│   │   ├── app.py
│   │   ├── routes/
│   │   └── middleware/
│   ├── services/           # Business Logic
│   │   ├── chat_service.py
│   │   ├── document_service.py
│   │   └── admin_service.py
│   ├── core/               # Configuration
│   │   ├── config.py
│   │   └── constants.py
│   └── utils/              # Utilities
│       ├── validators.py
│       └── helpers.py
├── migrations/             # Alembic DB migrations
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── package.json
├── desktop_app/            # Electron Desktop App
│   ├── main.js
│   ├── preload.js
│   └── package.json
├── requirements.txt        # Python dependencies
├── alembic.ini            # Database migration config
├── main.py                # Application entry point
└── README.md              # This file
```

### Data Flow Diagram

```
User Input
    ↓
FastAPI Endpoint
    ↓
LangGraph Agent
    ├→ Tool Router
    │   ├→ VnStock Tools (Stock data)
    │   ├→ Technical Tools (SMA, RSI)
    │   ├→ Financial Report Tools (OCR + AI)
    │   ├→ PDF Tools (Document parsing)
    │   └→ Excel Tools (Data analysis)
    ├→ LLM Provider
    │   ├→ Google Gemini (Cloud)
    │   └→ Ollama (Local)
    └→ Qdrant Vector DB (RAG retrieval)
    ↓
Markdown Response
    ↓
Frontend Display
```

### Database Schema

```sql
-- Users table
CREATE TABLE users (
  id UUID PRIMARY KEY,
  username VARCHAR UNIQUE,
  email VARCHAR UNIQUE,
  hashed_password VARCHAR,
  is_admin BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP
);

-- Chat sessions
CREATE TABLE chat_sessions (
  id UUID PRIMARY KEY,
  user_id UUID FOREIGN KEY,
  title VARCHAR,
  use_rag BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP
);

-- Chat messages
CREATE TABLE chat_messages (
  id UUID PRIMARY KEY,
  session_id UUID FOREIGN KEY,
  role VARCHAR,
  content TEXT,
  created_at TIMESTAMP
);

-- Document uploads
CREATE TABLE document_uploads (
  id UUID PRIMARY KEY,
  user_id UUID FOREIGN KEY,
  file_name VARCHAR,
  file_type VARCHAR,
  file_size INTEGER,
  created_at TIMESTAMP
);

-- Audit logs
CREATE TABLE audit_logs (
  id UUID PRIMARY KEY,
  user_id UUID FOREIGN KEY,
  action VARCHAR,
  timestamp TIMESTAMP
);
```

### Integration Points

**PostgreSQL ↔ FastAPI**
- SQLAlchemy ORM for data modeling
- Alembic for schema migrations
- Connection pooling for performance

**VnStock API ↔ Tools**
- Real-time stock prices
- Historical OHLCV data
- Company fundamentals
- Shareholder information

**LLM Providers ↔ Agent**
- Gemini: For analysis and OCR
- Ollama: For local chat
- Tool calling and function execution

**Qdrant ↔ RAG System**
- Vector embeddings storage
- Semantic document retrieval
- Collection management

---

## 📚 Learning Resources

### Official Documentation

- **VnStock**: https://vnstocks.com/docs/vnstock
- **LangChain**: https://python.langchain.com/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **FastAPI**: https://fastapi.tiangolo.com/
- **PostgreSQL**: https://www.postgresql.org/docs/
- **Qdrant**: https://qdrant.tech/documentation/

### Technical Analysis

- **TA-Lib Documentation**: https://ta-lib.org/
- **Investopedia**: https://www.investopedia.com/
- **Moving Averages**: https://investopedia.com/terms/m/movingaverage.asp
- **RSI Indicator**: https://investopedia.com/terms/r/rsi.asp

### Local LLM

- **Ollama**: https://ollama.com/
- **Ollama Models**: https://ollama.com/library
- **Ollama GitHub**: https://github.com/ollama/ollama

### AI/ML Frameworks

- **Google Gemini**: https://ai.google.dev/
- **LangChain**: https://www.langchain.com/
- **Hugging Face**: https://huggingface.co/

---

## 🚀 Deployment Guide

### Deploy Backend to Railway

1. Push code to GitHub
2. Connect GitHub repository to Railway
3. Set environment variables in Railway dashboard
4. Railway automatically detects Python and deploys

### Deploy Frontend to Vercel

1. Push frontend code to GitHub
2. Connect GitHub to Vercel
3. Configure build settings
4. Vercel auto-deploys on push

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t financial-agent .
docker run -p 8000:8000 --env-file .env financial-agent
```

---

## 🔄 Workflow Examples

### Example 1: Research a Stock

```
User: "Tell me everything about VNM stock"

Agent:
1. Uses get_company_info("VNM")
2. Uses get_shareholders("VNM")
3. Uses get_company_events("VNM")
4. Uses get_historical_data("VNM", period="6M")
5. Uses calculate_sma("VNM", window=20)
6. Uses calculate_rsi("VNM")
7. LLM synthesizes all data
8. Returns comprehensive analysis with tables
```

### Example 2: Financial Report Analysis

```
User: Upload financial report image

Agent:
1. OCR image → Extract text
2. Classify report type (Balance Sheet, Income Statement, etc.)
3. Extract financial tables → Markdown
4. Use Gemini to analyze data
5. Return formatted analysis with insights
```

### Example 3: Portfolio Analysis

```
User: "I own VNM, VCB, and HPG. How are they doing?"

Agent:
1. Gets latest data for each stock
2. Calculates technical indicators
3. Analyzes trends and momentum
4. Compares to market benchmarks
5. Provides investment insights
```

---

## 🌟 Features Roadmap

- [ ] Real-time price updates (WebSocket)
- [ ] Financial ratio calculations (P/E, ROE, ROA)
- [ ] News scraping and sentiment analysis
- [ ] Portfolio tracking and alerts
- [ ] Mobile app (React Native)
- [ ] Advanced charting and visualization
- [ ] Machine learning price predictions
- [ ] Multi-language support

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Create virtual environment
python -m venv venv_dev
source venv_dev/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8

# Run tests
pytest

# Format code
black src/

# Lint code
flake8 src/
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 👨‍💻 Author & Support

**Financial Agent** - AI Stock Market Assistant for Vietnam  
Built with ❤️ using modern AI and financial technologies

**Maintained by**: [Your Team/Name]  
**Project Status**: Active Development  
**Last Updated**: January 2025

### Quick Links

- 📧 **Email**: [contact@example.com]
- 💬 **Discussions**: GitHub Discussions
- 🐛 **Issues**: GitHub Issues
- 📖 **Wiki**: Project Wiki

### Acknowledgments

Special thanks to:
- VnStock team for the amazing free API
- LangChain team for the powerful framework
- Ollama team for local LLM support
- Google for Gemini API
- Open-source community

---

**Happy Trading! 📈🚀**

If you find this project helpful, please ⭐ star it on GitHub!
