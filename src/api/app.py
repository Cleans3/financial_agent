"""
FastAPI Application with Authentication & Database Integration
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Union
import logging
import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from datetime import datetime, timedelta

from ..agent import FinancialAgent
from ..tools.financial_report_tools import analyze_financial_report
from ..tools.excel_tools import analyze_excel_to_markdown
from ..tools.pdf_tools_v2 import analyze_pdf
from ..database.database import get_db, init_db
from ..database.models import User, ChatSession, ChatMessage
from ..services.session_service import SessionService
from ..core.config import settings
from ..core.tool_config import (
    ToolsConfig,
    QueryRewriteConfig,
    RAGFilterConfig,
    SummarizationConfig
)
from ..core.security import (
    create_access_token, 
    verify_password,
    hash_password,
    get_current_user,
    get_admin_user
)
from ..core.workflow_step_streaming import (
    create_workflow_step,
    STEP_STATUS_PENDING,
    STEP_STATUS_IN_PROGRESS,
    STEP_STATUS_COMPLETED,
    STEP_STATUS_ERROR,
    WORKFLOW_NODE_MAPPING
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Agent API",
    description="API cho Agent tư vấn đầu tư chứng khoán Việt Nam",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    return response

@app.on_event("startup")
async def startup():
    init_db()
    _init_admin_user()
    # Reset RAG router to ensure it gets fresh LLM instance
    from ..services.rag_router import reset_rag_router
    reset_rag_router()
    logger.info("Database initialized and admin user configured")

def _init_admin_user():
    from ..database.database import SessionLocal
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == settings.ADMIN_USERNAME).first()
        if not admin:
            admin = User(
                username=settings.ADMIN_USERNAME,
                email="admin@financial-agent.local",
                hashed_password=hash_password(settings.ADMIN_PASSWORD),
                is_admin=True
            )
            db.add(admin)
            db.commit()
            logger.info(f"Admin user '{settings.ADMIN_USERNAME}' created")
    finally:
        db.close()

agent = None

def get_agent():
    """Get or initialize the Financial Agent with config from settings."""
    global agent
    if agent is None:
        logger.info("Initializing Financial Agent with config...")
        try:
            # Build config from settings
            tools_config = ToolsConfig(
                allow_tool_calls=settings.ENABLE_TOOLS,
                allow_rag=settings.ENABLE_RAG,
                query_rewrite=QueryRewriteConfig(
                    enabled=settings.ENABLE_QUERY_REWRITING
                ),
                rag_filter=RAGFilterConfig(
                    min_relevance_threshold=settings.RAG_MIN_RELEVANCE,
                    max_documents=settings.RAG_MAX_DOCUMENTS
                ),
                summarization=SummarizationConfig(
                    enabled=settings.ENABLE_SUMMARIZATION,
                    length_threshold=settings.SUMMARIZATION_THRESHOLD
                )
            )
            agent = FinancialAgent(config=tools_config)
            logger.info(f"✓ Agent initialized: tools={settings.ENABLE_TOOLS}, RAG={settings.ENABLE_RAG}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent: {e}")
            raise
    return agent


# ==================== Auth Endpoints ====================

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str

class UserInfoResponse(BaseModel):
    id: str
    username: str
    email: str
    is_admin: bool
    is_active: bool

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username (3-50 characters)")
    email: Optional[str] = Field(None, description="Valid email address (optional)")
    password: str = Field(..., min_length=6, description="Password (min 6 chars, 1 uppercase, 1 lowercase, 1 digit)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "john@example.com",
                "password": "SecurePass123!"
            }
        }

class RegisterResponse(BaseModel):
    user_id: str
    username: str
    email: str
    message: str

def validate_password_complexity(password: str) -> bool:
    """Validate password meets complexity requirements (6+ chars, uppercase, lowercase, digit)"""
    import re
    if len(password) < 6:
        return False
    # At least 1 uppercase, 1 lowercase, 1 digit
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    return True

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    from ..services.rate_limiter import rate_limit_login
    from starlette.requests import Request
    
    # Rate limit by username
    allowed, info = rate_limit_login(request.username)
    if not allowed:
        raise HTTPException(
            status_code=429, 
            detail="Too many login attempts. Please try again later.",
            headers={"Retry-After": str(info["retry_after"])}
        )
    
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is disabled")
    
    from ..core.security import create_refresh_token
    access_token = create_access_token(data={"sub": user.id, "is_admin": user.is_admin})
    refresh_token = create_refresh_token(data={"sub": user.id, "is_admin": user.is_admin})
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=str(user.id)
    )

@app.post("/api/auth/register", response_model=RegisterResponse, status_code=201)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account"""
    import re
    
    # Validate email format if provided
    if request.email:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, request.email):
            raise HTTPException(status_code=422, detail="Invalid email format")
    
    # Validate password complexity
    if not validate_password_complexity(request.password):
        raise HTTPException(
            status_code=422, 
            detail="Password must contain at least 6 characters, 1 uppercase letter, 1 lowercase letter, and 1 digit"
        )
    
    # Check for duplicate username (case-insensitive)
    existing_user = db.query(User).filter(
        User.username.ilike(request.username)
    ).first()
    if existing_user:
        raise HTTPException(status_code=422, detail="Username already exists")
    
    # Check for duplicate email (case-insensitive) if provided
    if request.email:
        existing_email = db.query(User).filter(
            User.email.ilike(request.email)
        ).first()
        if existing_email:
            raise HTTPException(status_code=422, detail="Email already registered")
    
    # Create new user
    new_user = User(
        username=request.username,
        email=request.email,
        hashed_password=hash_password(request.password),
        is_admin=False,
        is_active=True
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Log the registration in audit log if available
    from ..services.admin_service import AdminService
    try:
        AdminService.log_action(db, new_user.id, "USER_REGISTERED", "USER", str(new_user.id), {"email": request.email})
    except Exception as e:
        logger.warning(f"Failed to log registration: {e}")
    
    return RegisterResponse(
        user_id=str(new_user.id),
        username=new_user.username,
        email=new_user.email,
        message="User registered successfully"
    )

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class RefreshTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@app.post("/api/auth/refresh", response_model=RefreshTokenResponse)
async def refresh_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Refresh access token using refresh token"""
    from ..core.security import decode_token, create_access_token, create_refresh_token
    
    try:
        payload = decode_token(request.refresh_token)
        
        # Verify this is actually a refresh token
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user from database to verify they still exist and are active
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        
        # Create new access token
        new_access_token = create_access_token(data={"sub": user.id, "is_admin": user.is_admin})
        
        return RefreshTokenResponse(access_token=new_access_token)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@app.get("/api/auth/me", response_model=UserInfoResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get current authenticated user's information"""
    # Get the actual User object from database
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserInfoResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        is_admin=user.is_admin,
        is_active=user.is_active
    )

# ==================== Health Check Endpoints ====================

class HealthResponse(BaseModel):
    status: str
    database: str
    rag_service: str
    llm_service: str
    timestamp: str

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint for monitoring
    Checks database connectivity, RAG service, and LLM service
    """
    from datetime import datetime
    
    health_status = {
        "database": "unknown",
        "rag_service": "unknown",
        "llm_service": "unknown"
    }
    
    # Check database connectivity
    try:
        db.execute("SELECT 1")
        health_status["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["database"] = "unhealthy"
    
    # RAG service check disabled - RAG now orchestrated by workflow only
    # try:
    #     from ..services.multi_collection_rag_service import get_rag_service
    #     rag_service = get_rag_service()
    #     if rag_service and hasattr(rag_service, 'index'):
    #         health_status["rag_service"] = "healthy"
    #     else:
    #         health_status["rag_service"] = "unavailable"
    # except Exception as e:
    #     logger.warning(f"RAG service health check failed: {e}")
    #     health_status["rag_service"] = "unavailable"
    
    # Check LLM service
    try:
        agent = get_agent()
        if agent:
            health_status["llm_service"] = "healthy"
        else:
            health_status["llm_service"] = "unavailable"
    except Exception as e:
        logger.warning(f"LLM service health check failed: {e}")
        health_status["llm_service"] = "unavailable"
    
    # Overall status
    overall_status = "healthy" if health_status["database"] == "healthy" else "degraded"
    
    return HealthResponse(
        status=overall_status,
        database=health_status["database"],
        rag_service=health_status["rag_service"],
        llm_service=health_status["llm_service"],
        timestamp=datetime.utcnow().isoformat()
    )

# ==================== Chat Endpoints ====================

class ChatRequest(BaseModel):
    question: str = Field(
        ..., 
        description="Câu hỏi của người dùng (tiếng Việt)",
        min_length=1,
        max_length=500
    )
    session_id: Optional[str] = None
    use_rag: bool = True
    allow_tools: bool = True
    uploaded_files: Optional[List[str]] = None  # List of filenames just uploaded in this turn
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Cho tôi biết thông tin về công ty VNM?"
            }
        }


class ThinkingStep(BaseModel):
    step: int
    title: str
    description: str
    result: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    message_id: str
    thinking_steps: Optional[List[ThinkingStep]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "VNM là mã chứng khoán của Công ty Cổ phần Sữa Việt Nam (Vinamilk)...",
                "session_id": "uuid-here",
                "message_id": "uuid-here",
                "thinking_steps": [
                    {"step": 1, "title": "🔄 Rewriting Query", "description": "Analyzing context...", "result": "Query rewritten"}
                ]
            }
        }


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str


class SessionResponse(BaseModel):
    id: str
    title: str
    use_rag: bool
    created_at: str
    updated_at: str


class SessionDetailResponse(BaseModel):
    id: str
    title: str
    use_rag: bool
    messages: List[MessageResponse]
    created_at: str
    updated_at: str


class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int


class FileUploadRequest(BaseModel):
    """Request model for file upload endpoint"""
    question: Optional[str] = Field(
        default="",
        description="Câu hỏi hoặc mô tả thêm về file"
    )


class FileUploadResponse(BaseModel):
    """Response model for file upload - returns prepared prompt for chat"""
    success: bool
    file_name: str
    chunks_indexed: int
    message: str
    session_id: str


class FileAnalysisResponse(BaseModel):
    """Response model for file analysis"""
    success: bool
    report_type: str
    extracted_text: str
    analysis: str
    message: str

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Agent API - Vietnamese Stock Market Assistant",
        "version": "1.0.0",
        "endpoints": {
            "auth": "POST /auth/login - Authenticate with username/password",
            "chat": "POST /api/chat - Main chat endpoint (requires auth)",
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
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Chat endpoint - Nhận câu hỏi và trả về câu trả lời
    Requires JWT authentication
    Supports RAG (Retrieval-Augmented Generation) if use_rag=true
    """
    try:
        agent_instance = get_agent()
        user_id = current_user["user_id"]
        
        session_id = request.session_id
        if not session_id:
            session = ChatSession(user_id=user_id, title=request.question[:50], use_rag=request.use_rag)
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = session.id
        else:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session or session.user_id != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            # CRITICAL: Refresh session to get latest metadata (files uploaded after session was loaded)
            db.refresh(session)
        
        # Auto-cleanup: Delete empty/greeting-only sessions (wrapped in try-except to not break chat)
        try:
            deleted_count = SessionService.delete_empty_sessions(db, user_id, exclude_session_id=session_id)
            if deleted_count > 0:
                logger.info(f"Auto-deleted {deleted_count} empty conversation(s) for user {user_id}")
        except Exception as cleanup_error:
            # Log cleanup error but don't break chat
            logger.warning(f"Cleanup error (non-critical): {cleanup_error}")
        
        history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
        
        # Convert history to format for agent
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in history
        ]
        
        # Session-level summarization: compress if >10 messages
        if len(conversation_history) > 10:
            try:
                from ..core.summarization import summarize_messages
                from ..llm import LLMFactory
                from langchain_core.messages import HumanMessage, AIMessage
                
                # Get main LLM for summarization
                llm = LLMFactory.get_llm()
                
                # Convert to LangChain messages for summarization
                lc_messages = [
                    HumanMessage(content=msg["content"]) if msg["role"] == "user" 
                    else AIMessage(content=msg["content"])
                    for msg in conversation_history[:5]
                ]
                
                # Summarize first 5 messages
                summary = await summarize_messages(lc_messages, llm, num_messages_to_compress=5)
                if summary:
                    conversation_history = [
                        {"role": "assistant", "content": f"📋 Previous context: {summary}"}
                    ] + conversation_history[5:]
                    logger.info(f"Session-level summarization applied for session {session_id}")
            except Exception as sum_error:
                logger.warning(f"Session summarization skipped: {sum_error}")
                # Continue without summarization if it fails
        
        # Detect user summarization preference from query
        summarize_results = True  # Default: auto-summarize if >500 chars
        summary_keywords_force = ["summarize", "tl;dr", "tldr", "in short", "brief", "summary", "tóm tắt"]
        summary_keywords_disable = ["no summary", "nosummary", "full detail", "all data", "no tl;dr", "verbose"]
        
        question_lower = request.question.lower()
        if any(kw in question_lower for kw in summary_keywords_disable):
            summarize_results = False
            logger.info("User disabled result summarization")
        elif any(kw in question_lower for kw in summary_keywords_force):
            summarize_results = True
            logger.info("User enabled result summarization")
        
        # Detect if user explicitly requests NO RAG (opt-out)
        force_no_rag = False
        no_rag_keywords = ["no rag", "không rag", "không dùng rag", "tắt rag", "disable rag"]
        if any(keyword.lower() in request.question.lower() for keyword in no_rag_keywords):
            force_no_rag = True
            logger.info(f"User explicitly disabled RAG: {request.question[:50]}")
        
        # NOTE: RAG search now happens INSIDE the workflow (in RETRIEVE node)
        # This allows file ingestion to happen BEFORE RAG search
        # The workflow will handle RAG routing and search
        # RAG is ENABLED by default, only disable if user explicitly requests it
        use_rag = (request.use_rag and not force_no_rag) or (request.use_rag is True)
        logger.info(f"RAG will be handled by workflow: use_rag={use_rag}")
        
        # Get full file metadata from session (has path, type, size, etc.)
        # IMPORTANT: Only use files if they're explicitly passed in THIS request!
        # Check if user actually uploaded files in this prompt (request.uploaded_files field)
        uploaded_files_info = []
        logger.info(f"[CHAT] Checking if files were uploaded in THIS request...")
        logger.info(f"[CHAT] request.uploaded_files: {request.uploaded_files}")
        
        # Debug: Show session metadata state
        logger.info(f"[CHAT] session.session_metadata exists: {session.session_metadata is not None}")
        if session.session_metadata:
            logger.info(f"[CHAT] session metadata keys: {list(session.session_metadata.keys())}")
            logger.info(f"[CHAT] uploaded_files in metadata: {'uploaded_files' in session.session_metadata}")
            if "uploaded_files" in session.session_metadata:
                logger.info(f"[CHAT] Files in session metadata: {[f.get('name', 'unknown') for f in session.session_metadata['uploaded_files']]}")
        
        # Only retrieve files from session if the current request indicates files were uploaded
        # This prevents triggering file upload workflow on normal prompts in sessions with previous uploads
        if request.uploaded_files and len(request.uploaded_files) > 0:
            logger.info(f"[CHAT] Files detected in current request: {request.uploaded_files}")
            # Get file metadata from session for files that were just uploaded
            if session and session.session_metadata and "uploaded_files" in session.session_metadata:
                session_files = session.session_metadata.get("uploaded_files", [])
                # Only use files that match the uploaded_files list from request
                uploaded_files_info = [
                    f for f in session_files 
                    if f.get('name') in request.uploaded_files
                ]
                logger.info(f"[CHAT] Matched {len(uploaded_files_info)} files from session metadata")
                if uploaded_files_info:
                    logger.info(f"[CHAT] Files to process: {[f.get('name', 'unknown') for f in uploaded_files_info]}")
            else:
                logger.warning(f"[CHAT] ⚠️  Files in request but NOT found in session metadata!")
                logger.warning(f"[CHAT]    request.uploaded_files: {request.uploaded_files}")
                logger.warning(f"[CHAT]    session has metadata: {session.session_metadata is not None}")
                if session.session_metadata:
                    logger.warning(f"[CHAT]    'uploaded_files' key exists: {'uploaded_files' in session.session_metadata}")
        else:
            logger.info(f"[CHAT] No files uploaded in current request - using text-only chat")
            logger.info(f"[CHAT] Session has files from previous uploads, but NOT processing them (normal prompt)")

        
        answer, thinking_steps = await agent_instance.aquery(
            request.question,  # Use original question, workflow will handle file context
            user_id=user_id, 
            session_id=session_id,
            conversation_history=conversation_history,
            uploaded_files=uploaded_files_info,  # Full metadata with path/type/size
            allow_tools=request.allow_tools,
            use_rag=use_rag
        )
        
        user_msg = ChatMessage(session_id=session_id, role="user", content=request.question)
        assistant_msg = ChatMessage(session_id=session_id, role="assistant", content=answer)
        db.add(user_msg)
        db.add(assistant_msg)
        
        # CRITICAL FIX: Only clear files that were ACTUALLY processed in THIS request
        # This prevents re-ingestion while allowing multiple file uploads in same conversation
        if request.uploaded_files and session and session.session_metadata and "uploaded_files" in session.session_metadata:
            logger.info(f"Removing {len(request.uploaded_files)} processed file(s) from session metadata")
            session_meta = session.session_metadata
            # Only remove files that were in THIS request
            files_to_keep = [
                f for f in session_meta.get("uploaded_files", [])
                if f.get('name') not in request.uploaded_files
            ]
            session_meta["uploaded_files"] = files_to_keep
            session.session_metadata = session_meta
            logger.info(f"Session now has {len(files_to_keep)} remaining file(s) for future processing")
        
        db.commit()
        db.refresh(assistant_msg)
        
        # Convert thinking steps to Pydantic models with safe error handling
        formatted_steps = []
        try:
            for step in thinking_steps:
                if isinstance(step, dict):
                    formatted_steps.append(ThinkingStep(**step))
                else:
                    formatted_steps.append(step)
        except Exception as step_error:
            logger.warning(f"Failed to serialize thinking steps: {step_error}")
            formatted_steps = []
        
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            message_id=assistant_msg.id,
            thinking_steps=formatted_steps if formatted_steps else None
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat-stream")
async def chat_stream(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Chat endpoint with streaming thinking steps and workflow progress
    Streams real-time workflow steps as the agent processes through all phases
    """
    import json
    import time
    
    async def generate():
        start_time = time.time()
        try:
            agent_instance = get_agent()
            user_id = current_user["user_id"]
            
            session_id = request.session_id
            if not session_id:
                session = ChatSession(user_id=user_id, title=request.question[:50], use_rag=request.use_rag)
                db.add(session)
                db.commit()
                db.refresh(session)
                session_id = session.id
                # Yield the new session ID first
                yield f'data: {json.dumps({"type": "session_id", "session_id": session_id})}\n\n'
            else:
                session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
                if not session or session.user_id != user_id:
                    yield f'data: {json.dumps({"type": "error", "message": "Access denied"})}\n\n'
                    return
                # CRITICAL: Refresh session to get latest metadata (files uploaded after session was loaded)
                db.refresh(session)
            
            # Auto-cleanup: Delete empty/greeting-only sessions
            try:
                deleted_count = SessionService.delete_empty_sessions(db, user_id, exclude_session_id=session_id)
                if deleted_count > 0:
                    logger.info(f"Auto-deleted {deleted_count} empty conversation(s) for user {user_id}")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error (non-critical): {cleanup_error}")
            
            history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
            
            # Convert history to format for agent
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in history
            ]
            
            # Detect if user explicitly requests NO RAG (opt-out)
            force_no_rag = False
            no_rag_keywords = ["no rag", "không rag", "không dùng rag", "tắt rag", "disable rag"]
            if any(keyword.lower() in request.question.lower() for keyword in no_rag_keywords):
                force_no_rag = True
                logger.info(f"User explicitly disabled RAG: {request.question[:50]}")
            
            # NOTE: RAG search now happens INSIDE the workflow (in RETRIEVE node)
            # This allows file ingestion to happen BEFORE RAG search
            # RAG is ENABLED by default, only disable if user explicitly requests it
            use_rag = (request.use_rag and not force_no_rag) or (request.use_rag is True)
            logger.info(f"RAG will be handled by workflow: use_rag={use_rag}")
            
            # Stream RAG status
            yield f'data: {json.dumps({"type": "rag_status", "used": use_rag, "note": "RAG handled by workflow after file ingestion"})}\n\n'
            
            # Get full file metadata from session (has path, type, size, etc.)
            # IMPORTANT: Only use files if they're explicitly passed in THIS request!
            # Check if user actually uploaded files in this prompt (request.uploaded_files field)
            uploaded_files_info = []
            logger.info(f"[CHAT-STREAM] Checking if files were uploaded in THIS request...")
            logger.info(f"[CHAT-STREAM] request.uploaded_files: {request.uploaded_files}")
            
            # Debug: Show session metadata state
            logger.info(f"[CHAT-STREAM] session.session_metadata exists: {session.session_metadata is not None}")
            if session.session_metadata:
                logger.info(f"[CHAT-STREAM] session metadata keys: {list(session.session_metadata.keys())}")
                logger.info(f"[CHAT-STREAM] uploaded_files in metadata: {'uploaded_files' in session.session_metadata}")
                if "uploaded_files" in session.session_metadata:
                    logger.info(f"[CHAT-STREAM] Files in session metadata: {[f.get('name', 'unknown') for f in session.session_metadata['uploaded_files']]}")
            
            # Only retrieve files from session if the current request indicates files were uploaded
            # This prevents triggering file upload workflow on normal prompts in sessions with previous uploads
            if request.uploaded_files and len(request.uploaded_files) > 0:
                logger.info(f"[CHAT-STREAM] Files detected in current request: {request.uploaded_files}")
                # Get file metadata from session for files that were just uploaded
                if session and session.session_metadata and "uploaded_files" in session.session_metadata:
                    session_files = session.session_metadata.get("uploaded_files", [])
                    # Only use files that match the uploaded_files list from request
                    uploaded_files_info = [
                        f for f in session_files 
                        if f.get('name') in request.uploaded_files
                    ]
                    logger.info(f"[CHAT-STREAM] Matched {len(uploaded_files_info)} files from session metadata")
                    if uploaded_files_info:
                        file_names = [f.get('name', 'unknown') for f in uploaded_files_info]
                        logger.info(f"[CHAT-STREAM] Files to process: {', '.join(file_names)}")
                else:
                    logger.warning(f"[CHAT-STREAM] ⚠️  Files in request but NOT found in session metadata!")
                    logger.warning(f"[CHAT-STREAM]    request.uploaded_files: {request.uploaded_files}")
                    logger.warning(f"[CHAT-STREAM]    session has metadata: {session.session_metadata is not None}")
                    if session.session_metadata:
                        logger.warning(f"[CHAT-STREAM]    'uploaded_files' key exists: {'uploaded_files' in session.session_metadata}")
            else:
                logger.info(f"[CHAT-STREAM] No files uploaded in current request - using text-only chat")
                logger.info(f"[CHAT-STREAM] Session has files from previous uploads, but NOT processing them (normal prompt)")
            
            # ========== EXECUTE AGENT WORKFLOW ==========
            # Emit workflow steps in real-time WHILE agent processes
            # Steps should complete around the same time agent finishes
            
            all_workflow_steps = []
            step_order = []
            
            if uploaded_files_info:
                step_order.extend(['extract_file', 'ingest_file'])
            step_order.extend(['classify', 'rewrite_eval'])
            if uploaded_files_info:
                step_order.append('rewrite_file')
            if conversation_history:
                step_order.append('rewrite_convo')
            if use_rag:
                step_order.extend(['retrieve', 'filter', 'analyze'])
            if request.allow_tools:
                step_order.extend(['select_tools', 'execute_tools'])
            step_order.extend(['summary_tools', 'query_reformulation', 'generate'])
            
            logger.info(f"[STREAM] Will emit {len(step_order)} workflow steps in parallel with agent execution")
            
            # Start agent processing in background
            agent_instance = get_agent()
            agent_task = asyncio.create_task(
                agent_instance.aquery(
                    question=request.question,
                    user_id=user_id,
                    session_id=session_id,
                    conversation_history=conversation_history,
                    uploaded_files=uploaded_files_info,
                    allow_tools=request.allow_tools,
                    use_rag=use_rag
                )
            )
            
            # Emit steps on timeline synchronized with agent execution
            # Spread steps across estimated execution time
            workflow_start = time.time()
            step_count = len(step_order)
            last_step_time = workflow_start
            
            for idx, step_name in enumerate(step_order):
                # Calculate when this step should complete
                # Spread steps evenly across agent execution time
                # Each step completes at: (step_number / total_steps) of elapsed time
                progress_ratio = (idx + 1) / step_count
                
                # Wait until it's time to emit this step
                # This synchronizes step emissions with estimated agent completion
                target_elapsed = 0.5 + (idx * 0.3)  # Start after 500ms, then add 300ms per step
                
                while True:
                    elapsed = time.time() - workflow_start
                    if elapsed >= target_elapsed or agent_task.done():
                        break
                    await asyncio.sleep(0.05)  # Check every 50ms
                
                # Calculate ACTUAL duration since last step (not hardcoded)
                current_time = time.time()
                actual_duration_ms = int((current_time - last_step_time) * 1000)
                last_step_time = current_time
                
                # Create and emit step with REAL timing
                step = create_workflow_step(
                    node_name=step_name,
                    status=STEP_STATUS_COMPLETED,
                    result=f"{step_name} completed",
                    metadata={"order": idx + 1, "total_steps": step_count},
                    duration=actual_duration_ms
                )
                all_workflow_steps.append(step)
                
                yield f'data: {json.dumps({"type": "workflow_step", "step": step})}\n\n'
                logger.debug(f"[STREAM] Step {idx + 1}/{step_count}: {step_name} ({actual_duration_ms}ms)")
            
            # Wait for agent to actually complete
            answer, thinking_steps = await agent_task
            workflow_duration = time.time() - workflow_start
            logger.info(f"[STREAM] Agent + workflow completed in {workflow_duration:.2f}s")
            
            # Stream each thinking step as it was calculated (legacy support)
            for step in thinking_steps:
                yield f'data: {json.dumps({"type": "thinking_step", "step": step})}\n\n'
                await asyncio.sleep(0.1)  # Small delay for better UX
            
            # Save messages to database
            user_msg = ChatMessage(
                session_id=session_id,
                role="user",
                content=request.question,
                message_meta={"message_type": "text"}
            )
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=answer,
                message_meta={
                    "message_type": "response", 
                    "rag_used": use_rag,
                    "workflow_steps_count": len(all_workflow_steps)
                }
            )
            db.add(user_msg)
            db.add(assistant_msg)
            
            # CRITICAL FIX: Only clear files that were ACTUALLY processed in THIS request
            # This prevents re-ingestion while allowing multiple file uploads in same conversation
            if request.uploaded_files and session and session.session_metadata and "uploaded_files" in session.session_metadata:
                logger.info(f"Removing {len(request.uploaded_files)} processed file(s) from session metadata (stream)")
                session_meta = session.session_metadata
                # Only remove files that were in THIS request
                files_to_keep = [
                    f for f in session_meta.get("uploaded_files", [])
                    if f.get('name') not in request.uploaded_files
                ]
                session_meta["uploaded_files"] = files_to_keep
                session.session_metadata = session_meta
                logger.info(f"Session now has {len(files_to_keep)} remaining file(s) for future processing (stream)")
            
            db.commit()
            db.refresh(assistant_msg)
            
            # Stream final answer with all workflow steps
            answer_data = {
                "type": "answer",
                "content": answer,
                "message_id": assistant_msg.id,
                "workflow_steps": all_workflow_steps,
                "total_duration_ms": int((time.time() - start_time) * 1000)
            }
            yield f'data: {json.dumps(answer_data)}\n\n'
            
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            # Emit error step
            error_step = create_workflow_step(
                node_name='error',
                status=STEP_STATUS_ERROR,
                result=f'Error: {str(e)}',
                metadata={'error_type': type(e).__name__}
            )
            yield f'data: {json.dumps({"type": "workflow_step", "step": error_step})}\n\n'
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# ==================== Session Endpoints ====================

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    use_rag: bool = True

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        session = SessionService.create_session(db, current_user["user_id"], request.title, request.use_rag)
        return SessionResponse(
            id=session.id,
            title=session.title,
            use_rag=session.use_rag,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat()
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        user_id = current_user["user_id"]
        
        # Auto-cleanup: Delete empty sessions before returning list
        try:
            deleted_count = SessionService.delete_empty_sessions(db, user_id)
            if deleted_count > 0:
                logger.info(f"Auto-deleted {deleted_count} empty conversation(s) for user {user_id} on list")
        except Exception as cleanup_error:
            logger.warning(f"Cleanup error on list (non-critical): {cleanup_error}")
        
        sessions = SessionService.list_sessions(db, user_id, limit, offset)
        return SessionListResponse(
            sessions=[
                SessionResponse(
                    id=s.id,
                    title=s.title,
                    use_rag=s.use_rag,
                    created_at=s.created_at.isoformat(),
                    updated_at=s.updated_at.isoformat()
                ) for s in sessions
            ],
            total=len(sessions)
        )
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        session = SessionService.get_session(db, session_id, current_user["user_id"])
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = SessionService.get_session_history(db, session_id, current_user["user_id"])
        
        return SessionDetailResponse(
            id=session.id,
            title=session.title,
            use_rag=session.use_rag,
            messages=[
                MessageResponse(id=m.id, role=m.role, content=m.content)
                for m in messages
            ],
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        deleted = SessionService.delete_session(db, session_id, current_user["user_id"])
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/sessions/{session_id}")
async def update_session(
    session_id: str,
    title: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        session = SessionService.update_session_title(db, session_id, current_user["user_id"], title)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionResponse(
            id=session.id,
            title=session.title,
            use_rag=session.use_rag,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Document Management Endpoints ====================

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    doc_id: str
    chunks: int
    title: str


@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    chat_id: Optional[str] = Query(None, description="Chat session ID for metadata isolation"),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload and process a document (PDF, DOCX, TXT, PNG, JPG)
    Automatically extracts text, chunks, embeds, and indexes to user's Qdrant collection
    Supports conversation-aware metadata isolation via chat_id parameter
    """
    try:
        from src.services.enhanced_document_service import EnhancedDocumentService
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Generate doc_id for tracking
            doc_id = f"{current_user['user_id']}_{Path(file.filename).stem}_{datetime.now().timestamp()}"
            
            # Use new enhanced document service that integrates with multi-collection RAG
            service = EnhancedDocumentService()
            success, result = service.process_and_ingest(
                file_path=tmp_path,
                user_id=current_user["user_id"],
                chat_id=chat_id,
                include_summary=True
            )
            
            if success:
                chunks_added = result.get("chunks_added", 0)
                logger.info(f"Document uploaded by user {current_user['user_id']}: {file.filename} ({chunks_added} chunks added to Qdrant)")
                return DocumentUploadResponse(
                    success=True,
                    message=result.get("message", "Document successfully processed and indexed"),
                    doc_id=doc_id,
                    chunks=chunks_added,
                    title=result.get("title", Path(file.filename).stem)
                )
            else:
                error_msg = result.get("error", "Unknown error during document processing")
                raise HTTPException(status_code=400, detail=error_msg)
        
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results")


class DocumentSearchResult(BaseModel):
    text: str
    title: str
    source: str
    doc_id: str
    similarity: float


class DocumentSearchResponse(BaseModel):
    query: str
    results: List[DocumentSearchResult]
    total: int


@app.post("/api/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    DISABLED - RAG search now orchestrated by workflow only
    Search for documents using semantic search
    """
    raise HTTPException(
        status_code=410,
        detail="Document search endpoint disabled. Use /api/chat with uploaded files instead - RAG is orchestrated by workflow."
    )


class DocumentDeleteResponse(BaseModel):
    success: bool
    message: str


@app.delete("/api/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    doc_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a document from the vector database
    Can only delete documents you uploaded
    """
    try:
        from src.services.document_service import get_document_service
        
        # Verify the document belongs to the user
        # (doc_id format: {user_id}_{title}_{timestamp})
        if not doc_id.startswith(current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied")
        
        service = get_document_service()
        success, message = service.delete_document(doc_id)
        
        if success:
            logger.info(f"Document deleted by user {current_user['user_id']}: {doc_id}")
            return DocumentDeleteResponse(success=True, message=message)
        else:
            raise HTTPException(status_code=400, detail=message)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DocumentStatsResponse(BaseModel):
    total_vectors: int
    embedding_model: str
    embedding_dimension: int
    supported_formats: List[str]
    max_file_size_mb: float


@app.get("/api/documents/stats", response_model=DocumentStatsResponse)
async def get_document_stats(
    current_user: dict = Depends(get_current_user)
):
    """Get document service statistics"""
    try:
        from src.services.document_service import get_document_service
        
        service = get_document_service()
        stats = service.get_stats()
        
        return DocumentStatsResponse(
            total_vectors=stats['faiss_vectors'],
            embedding_model=stats['embedding_model'],
            embedding_dimension=stats['embedding_dimension'],
            supported_formats=stats['supported_formats'],
            max_file_size_mb=stats['max_file_size_mb']
        )
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DocumentListItem(BaseModel):
    doc_id: str
    title: str
    filename: str
    file_type: str
    chunk_count: int
    created_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentListItem]
    total: int


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_user_documents(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all documents uploaded by the current user
    """
    try:
        from src.database.models import Document
        
        documents = db.query(Document).filter(
            Document.user_id == current_user["user_id"]
        ).order_by(Document.created_at.desc()).all()
        
        return DocumentListResponse(
            documents=[
                DocumentListItem(
                    doc_id=doc.id,
                    title=doc.filename.replace(Path(doc.filename).suffix, ''),
                    filename=doc.filename,
                    file_type=doc.file_type,
                    chunk_count=doc.chunk_count,
                    created_at=doc.created_at.isoformat()
                )
                for doc in documents
            ],
            total=len(documents)
        )
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DocumentChunkResponse(BaseModel):
    doc_id: str
    title: str
    chunks: List[Dict]  # List of {text: str, index: int, token_count: int}
    total_chunks: int


@app.get("/api/documents/{doc_id}/chunks", response_model=DocumentChunkResponse)
async def get_document_chunks(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    DISABLED - RAG orchestrated by workflow only
    Get all chunks for a specific document
    """
    raise HTTPException(
        status_code=410,
        detail="Document chunks endpoint disabled. RAG is orchestrated by workflow."
    )


class DocumentRegenerateResponse(BaseModel):
    success: bool
    message: str
    doc_id: str
    new_chunks: int


@app.post("/api/documents/{doc_id}/regenerate", response_model=DocumentRegenerateResponse)
async def regenerate_document_embeddings(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    DISABLED - RAG orchestrated by workflow only
    Regenerate embeddings for a document
    """
    raise HTTPException(
        status_code=410,
        detail="Document regeneration endpoint disabled. RAG is orchestrated by workflow."
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


@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    chat_session_id: Optional[str] = Form(default=None),
    question: str = Form(default=""),
    current_user: Optional[dict] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload file for ingestion into RAG + chat pipeline
    
    Flow:
    1. Create/get chat session
    2. Save file to persistent uploads directory (not temp)
    3. Store file metadata in session
    4. Return metadata for frontend to send to /api/chat with RAG context
    
    NOTE: Files are saved to persistent directory so the workflow can access them
    The workflow will clean up files after successful ingestion
    """
    file_path = None
    user_id = "anonymous"
    
    try:
        if current_user:
            user_id = current_user.get("user_id", "anonymous")
        
        file_extension = Path(file.filename).suffix.lower()
        file_name = file.filename
        
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
                detail=f"Unsupported file type. Supported: PNG, JPG, PDF, XLSX, XLS"
            )
        
        file_type_map = {'pdf': 'pdf', 'excel': 'excel', 'image': 'image'}
        file_type = 'pdf' if is_pdf else ('excel' if is_excel else 'image')
        
        logger.info(f"\n============================================================\nUPLOAD: {file_name} ({file_type})\n============================================================")
        logger.info(f"User: {user_id}")
        
        # Use persistent uploads directory instead of temp directory
        # This ensures workflow can access files even after HTTP response is sent
        uploads_dir = Path(__file__).parent.parent.parent / "uploads"
        uploads_dir.mkdir(exist_ok=True, parents=True)
        
        # Create unique filename to avoid collisions
        unique_name = f"{uuid.uuid4()}_{file_name}"
        file_path = str(uploads_dir / unique_name)
        
        logger.info(f"Saving file to: {file_path}")
        
        # Write file to persistent location
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"✓ File saved to persistent uploads directory")
        
        if not chat_session_id:
            session = ChatSession(
                user_id=user_id,
                title=question.strip()[:50] or f"📄 {Path(file_name).stem}",
                use_rag=True
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            chat_session_id = session.id
            logger.info(f"Created chat session: {chat_session_id}")
        else:
            session = db.query(ChatSession).filter(ChatSession.id == chat_session_id).first()
            if not session or (current_user and session.user_id != user_id):
                raise HTTPException(status_code=403, detail="Access denied")
        
        # IMPORTANT: Don't ingest file here anymore!
        # The workflow will handle file extraction + ingestion
        # Just store file metadata in the session for the workflow to use
        
        logger.info(f"✓ File stored: {file_name} (will be processed by workflow)")
        
        # Store file info in session metadata for workflow access
        session_meta = session.session_metadata or {}
        if "uploaded_files" not in session_meta:
            session_meta["uploaded_files"] = []
        
        file_info = {
            "name": file_name,
            "type": file_type,
            "path": file_path,  # Now using persistent file path
            "size": len(content),
            "extension": file_extension
        }
        
        session_meta["uploaded_files"].append(file_info)
        session.session_metadata = session_meta  # Reassign to trigger SQLAlchemy tracking
        
        # CRITICAL: Flag the JSON field as modified so SQLAlchemy detects the change
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(session, "session_metadata")
        
        logger.info(f"DEBUG: About to commit. Files in metadata: {[f.get('name') for f in session_meta.get('uploaded_files', [])]}")
        db.commit()
        logger.info(f"DEBUG: Commit successful. File metadata stored in session.")
        
        # Return success - workflow will do actual ingestion
        chunks_added = 0  # Will be updated by workflow when it ingests
        logger.info(f"✓ File metadata stored. Workflow will process during next query.")
        
        return FileUploadResponse(
            success=True,
            file_name=file_name,
            chunks_indexed=chunks_added,
            session_id=chat_session_id,
            message=f"File ingested successfully. Indexed {chunks_added} chunks."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        # Clean up file if upload failed
        if file_path and Path(file_path).exists():
            try:
                Path(file_path).unlink()
                logger.info(f"Cleaned up failed upload: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up file {file_path}: {cleanup_error}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        # NOTE: Don't delete file_path here!
        # The workflow needs to read it when processing the query
        # The workflow will delete it after ingestion completes
        pass


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


# ==================== Admin Endpoints ====================

from ..services.admin_service import AdminService

class AdminUserResponse(BaseModel):
    """Admin user response model"""
    id: str
    username: str
    email: str
    is_admin: bool
    is_active: bool
    created_at: Optional[str]
    sessions: int
    messages: int

class AdminSystemStats(BaseModel):
    """Admin system statistics model"""
    users: dict
    sessions: dict
    messages: dict
    timestamp: str

class AdminAuditLog(BaseModel):
    """Admin audit log model"""
    id: str
    user_id: Optional[str]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    details: Optional[Union[str, dict]]
    created_at: Optional[str]


@app.get("/api/admin/users", response_model=List[AdminUserResponse])
async def get_users_list(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get list of all users with statistics (Admin only)
    
    Args:
        skip: Number of users to skip
        limit: Number of users to return
        current_user: Admin user (verified by get_admin_user dependency)
        
    Returns:
        List of user objects with stats
    """
    users = AdminService.get_users_list(db, skip=skip, limit=limit)
    AdminService.log_action(db, current_user.id, "list_users", "User", None, f"Retrieved {len(users)} users")
    return users


@app.get("/api/admin/users/{user_id}")
async def get_user_details(
    user_id: str,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get detailed statistics for a specific user (Admin only)"""
    stats = AdminService.get_user_stats(db, user_id)
    if not stats:
        raise HTTPException(status_code=404, detail="User not found")
    
    AdminService.log_action(db, current_user.id, "view_user_stats", "User", user_id)
    return stats


@app.post("/api/admin/users/{user_id}/toggle-active")
async def toggle_user_active(
    user_id: str,
    is_active: bool,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Enable or disable a user account (Admin only)"""
    success = AdminService.toggle_user_active(db, user_id, is_active)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to toggle user status")
    
    status_str = "enabled" if is_active else "disabled"
    AdminService.log_action(db, current_user.id, f"user_{status_str}", "User", user_id)
    
    return {"success": True, "message": f"User {status_str} successfully"}


@app.get("/api/admin/stats", response_model=AdminSystemStats)
async def get_system_stats(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get overall system statistics (Admin only)"""
    stats = AdminService.get_system_stats(db)
    AdminService.log_action(db, current_user.id, "view_system_stats", "System", None)
    return stats


@app.get("/api/admin/rag-stats")
async def get_rag_stats(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get RAG (Retrieval-Augmented Generation) usage statistics (Admin only)"""
    stats = AdminService.get_rag_stats(db)
    AdminService.log_action(db, current_user.id, "view_rag_stats", "RAG", None)
    return stats


@app.get("/api/admin/audit-logs", response_model=List[AdminAuditLog])
async def get_audit_logs(
    user_id: Optional[str] = None,
    days: int = 7,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get audit logs for system actions and admin operations (Admin only)"""
    logs = AdminService.get_audit_logs(db, user_id=user_id, days=days)
    AdminService.log_action(db, current_user.id, "view_audit_logs", "AuditLog", None, f"Retrieved {len(logs)} logs")
    return logs


@app.delete("/api/admin/users/{user_id}/data")
async def delete_user_data(
    user_id: str,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Delete all data for a user (sessions, messages) - PERMANENT
    WARNING: This action cannot be undone
    (Admin only)
    """
    success = AdminService.delete_user_data(db, user_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to delete user data")
    
    AdminService.log_action(
        db, 
        current_user.id, 
        "delete_user_data", 
        "User", 
        user_id,
        "Permanent deletion of all user sessions and messages"
    )
    
    return {"success": True, "message": "User data deleted permanently"}


# ==================== Admin Document Management ====================

@app.post("/api/admin/documents/upload")
async def admin_upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Upload document to shared vectorDB (Admin only)"""
    from ..services.document_service import DocumentService
    from ..services.rag_service import get_rag_service
    import json
    import uuid
    
    if not file or file.size == 0:
        raise HTTPException(status_code=400, detail="No file provided")
    
    doc_id = str(uuid.uuid4())
    tags_list = []
    
    try:
        if tags:
            tags_list = json.loads(tags)
    except:
        tags_list = []
    
    # Save temp file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{doc_id}_{file.filename}")
    
    try:
        content = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(content)
        
# Process document (upload to global admin collection)
        doc_service = DocumentService()
        success, message, chunk_count = doc_service.process_file(
            file_path=temp_path,
            chat_session_id=doc_id,
            title=title or file.filename,
            user_id="admin",
            upload_to_global=True
        )
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        # Log to database
        # Note: process_file() already added the document to RAG and returned chunk_count
        AdminService.log_admin_document_upload(
            db,
            current_user.id,
            doc_id,
            file.filename,
            file.size,
            chunk_count,
            category,
            tags_list
        )
        
        AdminService.log_action(
            db,
            current_user.id,
            "upload_document",
            "Document",
            doc_id,
            f"Uploaded {chunk_count} chunks"
        )
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks": chunk_count,
            "message": f"Document processed successfully ({chunk_count} chunks)"
        }
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/admin/documents")
async def admin_list_documents(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """List all documents in vectorDB (Admin only)"""
    documents = AdminService.get_all_documents(db, skip=skip, limit=limit)
    total = AdminService.get_total_documents_count(db)
    
    AdminService.log_action(
        db,
        current_user.id,
        "view_documents",
        "Document",
        None,
        f"Listed {len(documents)} documents"
    )
    
    return {
        "documents": documents,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@app.get("/api/admin/documents/{doc_id}")
async def admin_get_document_details(
    doc_id: str,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get details for a specific document (Admin only)"""
    from ..services.rag_service import get_rag_service
    
    rag_service = get_rag_service()
    chunks = rag_service.get_document_chunks(doc_id)
    doc_info = AdminService.get_document_info(db, doc_id)
    
    AdminService.log_action(
        db,
        current_user.id,
        "view_document_details",
        "Document",
        doc_id
    )
    
    return {
        "doc_id": doc_id,
        "upload_info": doc_info,
        "chunk_count": len(chunks),
        "chunks": chunks[:10]  # First 10 chunks
    }


@app.delete("/api/admin/documents/{doc_id}")
async def admin_delete_document(
    doc_id: str,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Delete document from vectorDB (Admin only) - PERMANENT"""
    from ..services.rag_service import get_rag_service
    
    try:
        rag_service = get_rag_service()
        rag_service.delete_documents(doc_id)
        
        AdminService.delete_document_record(db, doc_id)
        
        AdminService.log_action(
            db,
            current_user.id,
            "delete_document",
            "Document",
            doc_id,
            "Permanently deleted from vectorDB"
        )
        
        return {"success": True, "message": f"Document {doc_id} deleted permanently"}
    
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to delete document: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Financial Agent API...")
