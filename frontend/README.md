# Financial Agent - React Frontend

Giao diện React hiện đại cho Financial Agent API.

## Tính năng

- ✨ UI/UX hiện đại, chuyên nghiệp
- 🎨 Dark theme với gradient đẹp mắt
- 💬 Chat interface mượt mà
- 📊 Hiển thị JSON data đẹp
- 📱 Responsive design
- ⚡ Real-time chat
- 🔍 Syntax highlighting cho code
- 📋 Copy to clipboard
- 🎯 Câu hỏi mẫu

## Cài đặt

```bash
cd frontend
npm install
```

## Chạy Development Server

```bash
npm run dev
```

Frontend sẽ chạy tại: http://localhost:3000

## Build Production

```bash
npm run build
npm run preview
```

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **TailwindCSS** - Styling
- **Axios** - HTTP client
- **React Markdown** - Render markdown
- **React Syntax Highlighter** - Code highlighting
- **Lucide React** - Icons

## API Integration

Frontend tự động proxy requests tới backend API (http://localhost:8000) qua Vite proxy.

Endpoint: `/api/chat`

## Cấu trúc

```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.jsx          # Header với logo
│   │   ├── Sidebar.jsx         # Sidebar với câu hỏi mẫu
│   │   ├── ChatInterface.jsx   # Main chat UI
│   │   └── MessageBubble.jsx   # Message component
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## Lưu ý

- Backend API phải chạy trước ở port 8000
- Proxy được cấu hình trong vite.config.js
