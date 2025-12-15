"""
FastAPI Application - REST API cho Financial Agent
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import asyncio
import os
import tempfile
from pathlib import Path

from ..agent import FinancialAgent
from ..tools.financial_report_tools import analyze_financial_report
from ..tools.excel_tools import analyze_excel_to_markdown
from ..tools.pdf_tools import analyze_pdf

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


class FileUploadRequest(BaseModel):
    """Request model for file upload endpoint"""
    question: Optional[str] = Field(
        default="",
        description="Câu hỏi hoặc mô tả thêm về file"
    )


class FileAnalysisResponse(BaseModel):
    """Response model for file analysis"""
    success: bool
    report_type: str
    extracted_text: str
    analysis: str
    message: str


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
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
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


@app.post("/api/upload", response_model=FileAnalysisResponse)
async def upload_file(
    file: UploadFile = File(...),
    question: str = Form(default="")
):
    """
    Upload và phân tích file (ảnh báo cáo tài chính hoặc Excel)
    
    Args:
        file: File ảnh (PNG, JPG, PDF) hoặc Excel (.xlsx, .xls)
        question: Câu hỏi hoặc mô tả thêm (optional)
        
    Returns:
        FileAnalysisResponse với kết QuảAnalysis
        
    Example:
        POST /api/upload
        Form data:
        - file: [binary file]
        - question: "Phân tích báo cáo tài chính này"
    """
    temp_path = None
    try:
        file_extension = Path(file.filename).suffix.lower()
        
        # Kiểm tra loại file
        image_types = ["image/png", "image/jpeg", "image/jpg"]
        pdf_types = ["application/pdf"]
        excel_types = [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "application/excel"
        ]
        
        is_image = file.content_type in image_types or file_extension in ['.png', '.jpg', '.jpeg']
        is_pdf = file.content_type in pdf_types or file_extension == '.pdf'
        is_excel = file.content_type in excel_types or file_extension in ['.xlsx', '.xls']
        
        if not (is_image or is_pdf or is_excel):
            raise HTTPException(
                status_code=400,
                detail=f"Loại file không được hỗ trợ. Hỗ trợ: PNG, JPG, PDF, XLSX, XLS. Nhận được: {file.content_type}"
            )
        
        # Lưu file tạm
        logger.info(f"Processing file: {file.filename}")
        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension,
            dir=tempfile.gettempdir()
        ) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
            logger.info(f"File saved to: {temp_path}")
        
        # Xử lý theo loại file
        if is_pdf:
            logger.info("Processing PDF file...")
            result = analyze_pdf(temp_path, question)
            
            if not result.success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Lỗi xử lý PDF: {result.message}"
                )
            
            # Kết hợp extracted text và tables markdown
            combined_data = f"Extracted Text:\n{result.extracted_text}\n\nTables:\n{result.tables_markdown}"
            
            # Gửi dữ liệu PDF cho Gemini phân tích nếu có câu hỏi
            analysis = result.analysis
            if question.strip():
                try:
                    from src.llm.llm_factory import LLMFactory
                    
                    logger.info("Sending PDF data to Gemini for analysis...")
                    
                    # Đọc system prompt
                    system_prompt_path = Path(__file__).parent.parent / "agent" / "prompts" / "financial_report_prompt.txt"
                    if system_prompt_path.exists():
                        with open(system_prompt_path, 'r', encoding='utf-8') as f:
                            base_system_prompt = f.read()
                    else:
                        base_system_prompt = "Bạn là một chuyên gia tài chính chuyên phân tích báo cáo tài chính doanh nghiệp. Hãy phân tích dữ liệu sau một cách chi tiết, chuyên sâu và đưa ra những insights quý giá cho nhà đầu tư."
                    
                    # Kết hợp system prompt + user question
                    system_prompt = f"{base_system_prompt}\n\n---\n\nYÊU CẦU TỪ NGƯỜI DÙNG:\n{question}"
                    
                    # Gọi Gemini
                    llm = LLMFactory.get_llm()
                    from langchain_core.messages import SystemMessage, HumanMessage
                    
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Phân tích dữ liệu PDF sau:\n\n{combined_data}")
                    ]
                    
                    response = await llm.ainvoke(messages)
                    analysis = response.content
                    
                    logger.info("✓ Gemini analysis completed for PDF")
                    
                except Exception as e:
                    logger.warning(f"Error sending PDF to Gemini: {e}")
                    analysis = result.analysis
            
            return FileAnalysisResponse(
                success=result.success,
                report_type=f"PDF Document ({result.processing_method.upper()})",
                extracted_text=result.extracted_text[:500] + "..." if len(result.extracted_text) > 500 else result.extracted_text,
                analysis=analysis,
                message=f"Xử lý PDF thành công. Method: {result.processing_method}"
            )
        
        elif is_excel:
            logger.info("Processing Excel file...")
            result = analyze_excel_to_markdown(temp_path)
            markdown_output = result.get('markdown', '')
            analysis = markdown_output
            
            # Gửi Markdown cho Gemini phân tích với system prompt tùy chỉnh
            try:
                from src.llm.llm_factory import LLMFactory
                
                logger.info("Sending Excel data to Gemini for analysis...")
                
                # Đọc system prompt mặc định từ file
                system_prompt_path = Path(__file__).parent.parent / "agent" / "prompts" / "excel_analysis_prompt.txt"
                if system_prompt_path.exists():
                    with open(system_prompt_path, 'r', encoding='utf-8') as f:
                        base_system_prompt = f.read()
                else:
                    base_system_prompt = "Bạn là một chuyên gia tài chính chuyên phân tích báo cáo tài chính doanh nghiệp. Hãy phân tích dữ liệu sau một cách chi tiết, chuyên sâu và đưa ra những insights quý giá cho nhà đầu tư."
                
                # Kết hợp system prompt mặc định + user instruction
                if question.strip():
                    system_prompt = f"{base_system_prompt}\n\n---\n\nYÊU CẦU TỪ NGƯỜI DÙNG:\n{question}"
                else:
                    system_prompt = base_system_prompt
                
                # Gọi Gemini trực tiếp
                llm = LLMFactory.get_llm()
                from langchain_core.messages import SystemMessage, HumanMessage
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Phân tích dữ liệu tài chính sau:\n\n{markdown_output}")
                ]
                
                response = await llm.ainvoke(messages)
                analysis = response.content
                
                logger.info("✓ Gemini analysis completed")
                
            except Exception as e:
                logger.error(f"Error sending to Gemini: {e}")
                analysis = markdown_output
            
            return FileAnalysisResponse(
                success=result['success'],
                report_type="Excel Financial Data",
                extracted_text=markdown_output,
                analysis=analysis,
                message=result['message']
            )
        
        else:  # is_image
            logger.info("Processing image file...")
            result = analyze_financial_report(temp_path)
            analysis = result.analysis
            extracted_text = result.extracted_text
            report_type = result.report_type
            
            try:
                from src.llm.llm_factory import LLMFactory
                from langchain_core.messages import SystemMessage, HumanMessage
                
                llm = LLMFactory.get_llm()
                
                # Determine how to process the image based on context
                if question.strip():
                    # User provided a specific question - use it as context
                    logger.info("User provided question - processing image with context...")
                    
                    # Combine the extracted text with the user question
                    # Let the agent decide how to handle it (financial analysis or regular question)
                    agent_instance = get_agent()
                    
                    # Combine extracted text with user question
                    combined_query = f"Tôi đã upload một ảnh có nội dung sau:\n\n{extracted_text}\n\n---\n\nCâu hỏi của tôi là: {question}"
                    
                    # Use the financial agent to handle the question
                    analysis = await agent_instance.aquery(combined_query)
                    
                    logger.info("✓ Agent processed image with user question")
                else:
                    # No user question provided - check if content looks like a financial report or a general question
                    logger.info("No question provided - analyzing image content type...")
                    
                    # Use LLM to determine content type and extract question if any
                    classification_prompt = f"""Analyze this extracted text from an image and determine:
1. Is this a financial report/statement? (yes/no)
2. If not, what type of content is it? (e.g., screenshot of question, chart, document, etc.)
3. If it contains a question, extract it.

Extracted text:
{extracted_text}

Respond in JSON format:
{{"is_financial_report": true/false, "content_type": "...", "question": "..." (if any)}}"""
                    
                    classification_response = await llm.ainvoke([
                        HumanMessage(content=classification_prompt)
                    ])
                    
                    try:
                        import json as json_module
                        classification = json_module.loads(classification_response.content)
                        is_financial = classification.get("is_financial_report", False)
                        extracted_question = classification.get("question", "")
                    except:
                        # Default to financial report if classification fails
                        is_financial = True
                        extracted_question = ""
                    
                    # Process based on content type
                    if is_financial:
                        # Process as financial report
                        logger.info("Content identified as financial report - using financial analysis")
                        system_prompt_path = Path(__file__).parent.parent / "agent" / "prompts" / "financial_report_prompt.txt"
                        if system_prompt_path.exists():
                            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                                base_system_prompt = f.read()
                        else:
                            base_system_prompt = "Bạn là một chuyên gia tài chính chuyên phân tích báo cáo tài chính doanh nghiệp. Hãy phân tích dữ liệu sau một cách chi tiết, chuyên sâu và đưa ra những insights quý giá cho nhà đầu tư."
                        
                        system_prompt = base_system_prompt
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=f"Phân tích dữ liệu từ báo cáo tài chính (đã OCR):\n\n{extracted_text}")
                        ]
                        response = await llm.ainvoke(messages)
                        analysis = response.content
                        report_type = "Financial Report (Image)"
                    else:
                        # Process as general question or content
                        logger.info("Content identified as non-financial - using general agent")
                        agent_instance = get_agent()
                        
                        if extracted_question:
                            # If a question was extracted, answer it with the context
                            combined_query = f"Dựa trên nội dung ảnh sau:\n\n{extracted_text}\n\n---\n\nCâu hỏi: {extracted_question}"
                            analysis = await agent_instance.aquery(combined_query)
                        else:
                            # Just provide information about the content
                            combined_query = f"Đây là nội dung từ ảnh được OCR:\n\n{extracted_text}\n\nHãy tóm tắt hoặc phân tích nội dung này."
                            analysis = await agent_instance.aquery(combined_query)
                        
                        report_type = f"Image ({classification.get('content_type', 'Unknown')})"
                    
                    logger.info("✓ Image processing completed")
                
            except Exception as e:
                logger.warning(f"Error in intelligent image processing: {e}")
                # Fallback to original analysis
                analysis = result.analysis
            
            return FileAnalysisResponse(
                success=result.success,
                report_type=report_type,
                extracted_text=extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                analysis=analysis,
                message=result.message
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý file: {str(e)}"
        )
    finally:
        # Xóa file tạm
        if temp_path:
            try:
                if os.path.exists(temp_path):
                    # Thử xóa file
                    os.remove(temp_path)
                    logger.info(f"Temp file deleted: {temp_path}")
            except PermissionError:
                # File đang được sử dụng - schedule xóa sau
                try:
                    import atexit
                    atexit.register(lambda: os.remove(temp_path) if os.path.exists(temp_path) else None)
                    logger.debug(f"Scheduled temp file cleanup: {temp_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Could not schedule cleanup: {cleanup_err}")
            except Exception as e:
                logger.warning(f"Error deleting temp file: {e}")


class ExcelAnalysisResponse(BaseModel):
    """Excel analysis response model"""
    success: bool
    file_name: str
    sheet_count: int
    sheet_names: List[str]
    markdown: str
    analysis: Optional[str] = None
    message: str


@app.post("/api/upload-excel", response_model=ExcelAnalysisResponse)
async def upload_excel_file(
    file: UploadFile = File(...),
    question: str = Form(default="")
):
    """
    Upload và phân tích file Excel (dữ liệu tài chính)
    
    Args:
        file: File Excel (.xlsx, .xls)
        question: Câu hỏi hoặc mô tả thêm (optional)
        
    Returns:
        ExcelAnalysisResponse với Markdown và phân tích
        
    Example:
        POST /api/upload-excel
        Form data:
        - file: [binary excel file]
        - question: "Phân tích báo cáo tài chính này"
    """
    temp_path = None
    try:
        # Kiểm tra loại file
        allowed_types = [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "application/excel"
        ]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Loại file không được hỗ trợ. Hỗ trợ: .xlsx, .xls. Nhận được: {file.content_type}"
            )
        
        # Lưu file tạm
        logger.info(f"Processing Excel file: {file.filename}")
        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(file.filename).suffix,
            dir=tempfile.gettempdir()
        ) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
            logger.info(f"File saved to: {temp_path}")
        
        # Phân tích Excel thành Markdown
        logger.info("Converting Excel to Markdown...")
        result = analyze_excel_to_markdown(temp_path)
        
        # Nếu có câu hỏi, gửi Markdown cho Gemini/Agent xử lý
        analysis = ""
        if question.strip():
            try:
                agent_instance = get_agent()
                combined_question = f"{question}\n\nDữ liệu tài chính (Markdown):\n\n{result['markdown']}"
                answer = await agent_instance.aquery(combined_question)
                analysis = answer
                logger.info("✓ Agent analysis completed")
            except Exception as e:
                logger.warning(f"Error sending to agent: {e}")
                analysis = ""
        
        return ExcelAnalysisResponse(
            success=result['success'],
            file_name=result['file_name'],
            sheet_count=result['sheet_count'],
            sheet_names=result['sheet_names'],
            markdown=result['markdown'],
            analysis=analysis,
            message=result['message']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý file Excel: {str(e)}"
        )
    finally:
        # Xóa file tạm
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Temp file deleted: {temp_path}")
            except Exception as e:
                logger.warning(f"Error deleting temp file: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Financial Agent API...")
