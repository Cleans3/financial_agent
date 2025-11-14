"""
REST API for Financial Agent - Vietnamese Stock Market
FastAPI application exposing /ask endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.financial_agent import FinancialAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Agent API - Vietnamese Stock Market",
    description="Agent hỗ trợ tra cứu thông tin chứng khoán Việt Nam",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent (singleton)
agent: FinancialAgent = None


class AskRequest(BaseModel):
    """Request model for /ask endpoint"""
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Thông tin về công ty VNM"
            }
        }


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""
    answer: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Công ty VNM (Vinamilk) là..."
            }
        }


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent
    try:
        logger.info("Initializing Financial Agent...")
        agent = FinancialAgent()
        logger.info("Financial Agent initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "Financial Agent API - Vietnamese Stock Market",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "POST - Hỏi câu hỏi về chứng khoán",
            "/health": "GET - Kiểm tra health",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_ready": agent is not None
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Xử lý câu hỏi về chứng khoán Việt Nam
    
    Hỗ trợ:
    - Tra cứu thông tin công ty theo mã chứng khoán
    - Lấy dữ liệu giá lịch sử (OHLCV)
    - Tính toán chỉ số kỹ thuật (SMA, RSI)
    
    Args:
        request: AskRequest với câu hỏi tiếng Việt
        
    Returns:
        AskResponse với câu trả lời
        
    Examples:
        - "Thông tin về công ty VNM"
        - "Dữ liệu giá VCB 3 tháng gần nhất"
        - "Tính SMA-20 cho HPG từ 2023-01-01 đến 2023-06-30"
        - "RSI của VIC trong 1 tháng gần nhất"
    """
    try:
        if not agent:
            raise HTTPException(
                status_code=500,
                detail="Agent chưa được khởi tạo"
            )
        
        question = request.question.strip()
        if not question:
            raise HTTPException(
                status_code=400,
                detail="Câu hỏi không được để trống"
            )
        
        logger.info(f"Processing question: {question}")
        
        # Query agent
        answer = await agent.aquery(question)
        
        logger.info(f"Answer generated successfully")
        
        return AskResponse(answer=answer)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý câu hỏi: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
