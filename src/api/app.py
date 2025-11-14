"""
FastAPI Application - REST API cho Financial Agent
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
import asyncio

from ..agent import FinancialAgent

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Financial Agent API",
    description="API cho Agent tư vấn đầu tư chứng khoán Việt Nam",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent (lazy loading)
agent = None


def get_agent():
    """Get or create agent instance"""
    global agent
    if agent is None:
        logger.info("Initializing Financial Agent...")
        agent = FinancialAgent()
    return agent


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str = Field(
        ..., 
        description="Câu hỏi của người dùng (tiếng Việt)",
        min_length=1,
        max_length=500
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Cho tôi biết thông tin về công ty VNM?"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="Câu trả lời từ Agent")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "VNM là mã chứng khoán của Công ty Cổ phần Sữa Việt Nam (Vinamilk)..."
            }
        }


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Agent API - Vietnamese Stock Market Assistant",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /api/chat - Main chat endpoint",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation (Swagger UI)",
            "redoc": "GET /redoc - API documentation (ReDoc)"
        },
        "capabilities": [
            "Tra cứu thông tin công ty (get_company_info)",
            "Dữ liệu giá lịch sử (get_historical_data)",
            "Tính SMA - Simple Moving Average (calculate_sma)",
            "Tính RSI - Relative Strength Index (calculate_rsi)"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if agent can be initialized
        agent_instance = get_agent()
        return {
            "status": "healthy",
            "agent_ready": agent_instance is not None,
            "tools_count": len(agent_instance.tools) if agent_instance else 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - Nhận câu hỏi và trả về câu trả lời
    
    Args:
        request: ChatRequest với field 'question'
        
    Returns:
        ChatResponse với field 'answer'
        
    Example:
        POST /api/chat
        {
            "question": "Thông tin về công ty VNM?"
        }
        
        Response:
        {
            "answer": "VNM là Công ty Cổ phần Sữa Việt Nam..."
        }
    """
    try:
        logger.info(f"Received question: {request.question}")
        
        # Get agent instance
        agent_instance = get_agent()
        
        # Process question using agent
        answer = await agent_instance.aquery(request.question)
        
        logger.info(f"Answer generated successfully (length: {len(answer)})")
        
        return ChatResponse(answer=answer)
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Lỗi xử lý câu hỏi: {str(e)}"
        )


@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "status": "ok",
        "message": "API đang hoạt động bình thường",
        "sample_questions": [
            "Cho tôi biết thông tin về công ty VNM?",
            "Giá VCB trong 3 tháng gần nhất?",
            "Tính SMA 20 ngày cho HPG",
            "RSI của VIC hiện tại?"
        ],
        "usage": {
            "endpoint": "POST /api/chat",
            "body": {
                "question": "your question here"
            }
        }
    }


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 60)
    logger.info("Financial Agent API Starting...")
    logger.info("=" * 60)
    
    # Pre-initialize agent
    try:
        get_agent()
        logger.info("✅ Agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
    
    logger.info("🚀 API ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Financial Agent API...")
