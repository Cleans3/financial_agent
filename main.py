"""
Main Entry Point - Chạy FastAPI server
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get config from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Financial Agent API                             ║
║       Vietnamese Stock Market Investment Assistant           ║
╚══════════════════════════════════════════════════════════════╝

🚀 Starting server...
📍 API Server: http://{host}:{port}
📚 API Documentation (Swagger UI): http://{host}:{port}/docs
📖 API Documentation (ReDoc): http://{host}:{port}/redoc
🏥 Health Check: http://{host}:{port}/health

💡 Features:
   ✅ Tra cứu thông tin công ty
   ✅ Dữ liệu giá lịch sử (OHLCV)
   ✅ Phân tích kỹ thuật (SMA, RSI)
   ✅ Hỗ trợ tiếng Việt

Press CTRL+C to quit
""")
    
    # Run FastAPI server
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
