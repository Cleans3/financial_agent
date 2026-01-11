"""
VnStock Tools - Công cụ tra cứu thông tin chứng khoán Việt Nam
Sử dụng VnStock API (Free) để lấy dữ liệu
"""

import json
import re
import logging
from typing import Optional
from datetime import datetime, timedelta
from langchain_core.tools import tool
from pydantic import BaseModel, Field, validator
import pandas as pd
import sys
import io

# Suppress vnstock promotional output that causes Unicode encoding errors on Windows
# vnstock tries to print emoji which cp1252 encoding can't handle
_old_stdout = None
try:
    _old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    from vnstock import Vnstock
    sys.stdout = _old_stdout
except Exception as e:
    if _old_stdout:
        sys.stdout = _old_stdout
    # If vnstock import fails, log warning but continue
    logging.warning(f"Failed to import vnstock: {e}")
    Vnstock = None

logger = logging.getLogger(__name__)


# ============================================
# Pydantic Models cho Tool Input Validation
# ============================================

class CompanyInfoInput(BaseModel):
    """Input schema for company info tool"""
    ticker: str = Field(
        description="Mã chứng khoán viết HOA (VD: VNM, VCB, HPG, FPT, VIC)"
    )
    
    @validator('ticker')
    def validate_ticker(cls, value):
        """Validate ticker format"""
        if not value:
            raise ValueError("Ticker không được để trống")
        # Convert to uppercase
        value = value.upper().strip()
        # Check if alphanumeric and 3-4 characters
        if not re.match(r'^[A-Z]{3,4}$', value):
            raise ValueError("Ticker phải là 3-4 ký tự chữ cái (VD: VNM, VCB)")
        return value


class HistoricalDataInput(BaseModel):
    """Input schema for historical data tool"""
    ticker: str = Field(
        description="Mã chứng khoán viết HOA (VD: VNM, VCB, HPG)"
    )
    start_date: Optional[str] = Field(
        None,
        description="Ngày bắt đầu format YYYY-MM-DD (VD: 2023-01-01)"
    )
    end_date: Optional[str] = Field(
        None,
        description="Ngày kết thúc format YYYY-MM-DD (VD: 2023-06-30)"
    )
    period: Optional[str] = Field(
        None,
        description="Khoảng thời gian: 1M (1 tháng), 3M (3 tháng), 6M (6 tháng), 1Y (1 năm)"
    )
    
    @validator('ticker')
    def validate_ticker(cls, value):
        if not value:
            raise ValueError("Ticker không được để trống")
        return value.upper().strip()
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, value):
        """Validate date format YYYY-MM-DD"""
        if value is None:
            return value
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, value):
            raise ValueError("Ngày phải có format YYYY-MM-DD (VD: 2023-01-01)")
        # Try parsing to validate it's a real date
        try:
            datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Ngày không hợp lệ: {value}")
        return value
    
    @validator('period')
    def validate_period(cls, value):
        """Validate period format"""
        if value is None:
            return value
        pattern = r'^\d+[MDWY]$'  # 1M, 3M, 6M, 1Y, etc.
        if not re.match(pattern, value.upper()):
            raise ValueError("Period phải có format như: 1M, 3M, 6M, 1Y")
        return value.upper()


# ============================================
# Tool Implementations
# ============================================

@tool("get_company_info")
def get_company_info(ticker: str) -> str:
    """
    Lấy thông tin chi tiết về công ty theo mã chứng khoán.
    
    Args:
        ticker: Mã chứng khoán viết HOA (VD: VNM, VCB, HPG, FPT, VIC)
        
    Returns:
        JSON string chứa thông tin công ty
    """
    try:
        # Handle both direct string and wrapped dict arguments
        if isinstance(ticker, dict):
            if 'parameters' in ticker:
                ticker = ticker['parameters'].get('ticker', '')
            elif 'ticker' in ticker:
                ticker = ticker['ticker']
            else:
                ticker = list(ticker.values())[0] if ticker else ''
        
        # Validate and normalize ticker
        ticker = str(ticker).upper().strip()
        
        logger.info("="*30)
        logger.info(f"🔍 SEARCHING: get_company_info")
        logger.info(f"   Ticker: {ticker}")
        logger.info("="*30)
        
        # Validate ticker format
        if not ticker or len(ticker) < 3:
            error_msg = "Mã chứng khoán không hợp lệ. Vui lòng nhập mã 3-4 ký tự."
            logger.warning(f"❌ Invalid ticker: {ticker}")
            return json.dumps({
                "success": False,
                "message": error_msg
            }, ensure_ascii=False)
        
        ticker = ticker.upper().strip()
        
        # Initialize VnStock
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        
        # Get company overview
        company_info = stock.company.overview()
        
        if company_info is None or (isinstance(company_info, pd.DataFrame) and company_info.empty):
            error_msg = f"Không tìm thấy thông tin cho mã {ticker}. Vui lòng kiểm tra lại mã chứng khoán."
            logger.warning(f"❌ Company not found: {ticker}")
            return json.dumps({
                "success": False,
                "message": error_msg,
                "data": None
            }, ensure_ascii=False)
        
        # Convert to dict
        if isinstance(company_info, pd.DataFrame):
            info = company_info.to_dict('records')[0]
        else:
            info = company_info
        
        # Extract company name from company_profile (first sentence usually contains it)
        company_profile = info.get('company_profile', '')
        company_name = company_profile.split('.')[0] if company_profile else 'N/A'
        
        # Format response with correct field mapping
        result = {
            "success": True,
            "ticker": ticker,
            "data": {
                "company_name": company_name,
                "symbol": info.get('symbol', ticker),
                "exchange": "HOSE",  # VNM is on HOSE, you could add logic to determine this
                "industry": info.get('icb_name4', info.get('icb_name3', 'N/A')),
                "industry_group": info.get('icb_name2', 'N/A'),
                "charter_capital": info.get('charter_capital', 'N/A'),
                "issue_share": info.get('issue_share', 'N/A'),
                "company_profile": company_profile[:500] + '...' if len(company_profile) > 500 else company_profile,
                "history": info.get('history', 'N/A')[:300] + '...' if info.get('history') and len(info.get('history', '')) > 300 else info.get('history', 'N/A'),
            },
            "message": f"Đã tìm thấy thông tin công ty {ticker}"
        }
        
        logger.info(f"✓ Search result:")
        logger.info(f"  Company: {company_name}")
        logger.info(f"  Industry: {result['data']['industry']}")
        logger.info(f"  Profile: {result['data']['company_profile'][:100]}...")
        logger.info(f"Full Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
        logger.info("="*30)
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        logger.info("="*30)
        return json.dumps({
            "success": False,
            "message": f"Lỗi khi lấy thông tin công ty {ticker}: {str(e)}",
            "data": None
        }, ensure_ascii=False)


@tool("get_historical_data")
def get_historical_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None
) -> str:
    """
    Lấy dữ liệu giá lịch sử (OHLCV) của mã chứng khoán.
    
    Args:
        ticker: Mã chứng khoán viết HOA (VD: VNM, VCB, HPG)
        start_date: Ngày bắt đầu format YYYY-MM-DD (VD: "2023-01-01")
        end_date: Ngày kết thúc format YYYY-MM-DD (VD: "2023-06-30")
        period: Khoảng thời gian (VD: "3M" = 3 tháng, "6M" = 6 tháng, "1Y" = 1 năm)
        
    Returns:
        JSON string chứa dữ liệu giá lịch sử và thống kê
    """
    try:
        # Handle both direct string and wrapped dict arguments
        if isinstance(ticker, dict):
            if 'parameters' in ticker:
                params = ticker['parameters']
                ticker = params.get('ticker', '')
                start_date = params.get('start_date', start_date)
                end_date = params.get('end_date', end_date)
                period = params.get('period', period)
            elif 'ticker' in ticker:
                ticker = ticker['ticker']
        
        ticker = str(ticker).upper().strip()
        
        logger.info("="*30)
        logger.info(f"🔍 SEARCHING: get_historical_data")
        logger.info(f"   Ticker: {ticker}")
        logger.info(f"   Start: {start_date}, End: {end_date}, Period: {period}")
        logger.info("="*30)
        
        if not ticker or len(ticker) < 3:
            logger.warning(f"❌ Invalid ticker: {ticker}")
            return json.dumps({
                "success": False,
                "message": "Mã chứng khoán không hợp lệ."
            }, ensure_ascii=False)
        
        # Calculate dates based on period if provided
        if period and not start_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Handle Vietnamese text and normalize period
            period_normalized = str(period).strip().lower()
            
            # Map Vietnamese descriptions to standard format
            vietnamese_to_period = {
                '3 tháng gần nhất': '3M',
                '3 tháng': '3M',
                '6 tháng gần nhất': '6M',
                '6 tháng': '6M',
                '1 năm gần nhất': '1Y',
                '1 năm': '1Y',
                'năm qua': '1Y',
                '2 năm': '2Y',
                '5 năm': '5Y',
                '1 tháng': '1M',
                '2 tháng': '2M',
                '1 tuần': '1W',
                '2 tuần': '2W',
                '1 ngày': '1D',
                '1 tháng gần nhất': '1M',
                '2 tháng gần nhất': '2M',
            }
            
            # Try to match Vietnamese period
            if period_normalized in vietnamese_to_period:
                period_normalized = vietnamese_to_period[period_normalized]
            
            # Extract numeric value and unit from normalized period
            try:
                period_value = int(period_normalized[:-1])
                period_unit = period_normalized[-1].upper()
            except (ValueError, IndexError):
                logger.warning(f"❌ Invalid period format: {period}. Using default 3 months.")
                period_value = 3
                period_unit = 'M'
            
            if period_unit == 'M':
                start_date = (datetime.now() - timedelta(days=period_value * 30)).strftime("%Y-%m-%d")
            elif period_unit == 'Y':
                start_date = (datetime.now() - timedelta(days=period_value * 365)).strftime("%Y-%m-%d")
            elif period_unit == 'D':
                start_date = (datetime.now() - timedelta(days=period_value)).strftime("%Y-%m-%d")
            elif period_unit == 'W':
                start_date = (datetime.now() - timedelta(weeks=period_value)).strftime("%Y-%m-%d")
        
        # Default: 3 months if no dates provided
        if not start_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get historical data from VnStock
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        df = stock.quote.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            logger.warning(f"❌ No data found for {ticker}")
            logger.info("="*30)
            return json.dumps({
                "success": False,
                "message": f"Không có dữ liệu giá cho {ticker} từ {start_date} đến {end_date}",
                "data": None
            }, ensure_ascii=False)
        
        # Calculate statistics
        stats = {
            "total_records": len(df),
            "highest_price": float(df['high'].max()),
            "lowest_price": float(df['low'].min()),
            "avg_close": float(df['close'].mean()),
            "avg_volume": int(df['volume'].mean()),
            "latest_close": float(df['close'].iloc[-1]),
            "first_close": float(df['close'].iloc[0]),
            "price_change": float(df['close'].iloc[-1] - df['close'].iloc[0]),
            "price_change_percent": float(((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100)
        }
        
        # Get ALL data for detailed table display
        detailed_data = []
        for idx, row in df.iterrows():
            detailed_data.append({
                "date": row['time'].strftime('%Y-%m-%d') if isinstance(row['time'], pd.Timestamp) else str(row['time']),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            })
        
        result = {
            "success": True,
            "ticker": ticker,
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "statistics": stats,
            "detailed_data": detailed_data,  # Full data for table display
            "message": f"Đã lấy được {len(df)} bản ghi dữ liệu giá cho {ticker}. Hiển thị chi tiết trong bảng."
        }
        
        logger.info(f"✓ Search result:")
        logger.info(f"  Records: {len(df)}")
        logger.info(f"  Period: {start_date} to {end_date}")
        logger.info(f"  Statistics: High={stats['highest_price']}, Low={stats['lowest_price']}, Change={stats['price_change_percent']:.2f}%")
        logger.info(f"  Data rows: {len(detailed_data)}")
        logger.info(f"Full Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
        logger.info("="*30)
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        logger.info("="*30)
        return json.dumps({
            "success": False,
            "message": f"Lỗi khi lấy dữ liệu giá {ticker}: {str(e)}",
            "data": None
        }, ensure_ascii=False)


@tool("get_shareholders")
def get_shareholders(ticker: str) -> str:
    """
    Lấy danh sách cổ đông lớn của công ty.
    
    Args:
        ticker: Mã chứng khoán viết HOA (VD: VNM, VCB, HPG)
        
    Returns:
        JSON string chứa danh sách cổ đông
    """
    try:
        # Handle both direct string and wrapped dict arguments
        if isinstance(ticker, dict):
            if 'parameters' in ticker:
                ticker = ticker['parameters'].get('ticker', '')
            elif 'ticker' in ticker:
                ticker = ticker['ticker']
        
        ticker = str(ticker).upper().strip()
        
        if not ticker or len(ticker) < 3:
            return json.dumps({
                "success": False,
                "message": "Mã chứng khoán không hợp lệ."
            }, ensure_ascii=False)
        
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        shareholders = stock.company.shareholders()
        
        if shareholders is None or (isinstance(shareholders, pd.DataFrame) and shareholders.empty):
            return json.dumps({
                "success": False,
                "message": f"Không tìm thấy thông tin cổ đông cho mã {ticker}",
                "data": None
            }, ensure_ascii=False)
        
        # Get top 10 shareholders
        top_shareholders = shareholders.head(10).to_dict('records')
        
        # Format data
        formatted_shareholders = []
        for sh in top_shareholders:
            formatted_shareholders.append({
                "name": sh.get('share_holder', 'N/A'),
                "quantity": int(sh.get('quantity', 0)),
                "ownership_percent": float(sh.get('share_own_percent', 0)) * 100,
                "update_date": str(sh.get('update_date', 'N/A'))
            })
        
        result = {
            "success": True,
            "ticker": ticker,
            "total_shareholders": len(shareholders),
            "top_shareholders": formatted_shareholders,
            "message": f"Đã tìm thấy {len(shareholders)} cổ đông của {ticker}"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Lỗi khi lấy thông tin cổ đông {ticker}: {str(e)}",
            "data": None
        }, ensure_ascii=False)


@tool("get_officers")
def get_officers(ticker: str) -> str:
    """
    Lấy danh sách ban lãnh đạo của công ty.
    
    Args:
        ticker: Mã chứng khoán viết HOA (VD: VNM, VCB, HPG)
        
    Returns:
        JSON string chứa danh sách ban lãnh đạo
    """
    try:
        # Handle both direct string and wrapped dict arguments
        if isinstance(ticker, dict):
            if 'parameters' in ticker:
                ticker = ticker['parameters'].get('ticker', '')
            elif 'ticker' in ticker:
                ticker = ticker['ticker']
        
        # Validate ticker input
        if not ticker or not isinstance(ticker, str):
            raise ValueError(f"Ticker phải là chuỗi không rỗng, nhận được: {type(ticker)} = {ticker}")
        
        ticker = str(ticker).upper().strip()
        
        if not ticker or len(ticker) < 3:
            raise ValueError(f"Ticker phải có ít nhất 3 ký tự, nhận được: '{ticker}'")
        
        # Initialize VnStock
        try:
            stock = Vnstock().stock(symbol=ticker, source='VCI')
        except Exception as e:
            return json.dumps({
                "success": False,
                "message": f"Lỗi kết nối VnStock cho mã {ticker}: {str(e)}",
                "data": None
            }, ensure_ascii=False)
        
        # Get officers data
        officers = stock.company.officers()
        
        if officers is None or (isinstance(officers, pd.DataFrame) and officers.empty):
            return json.dumps({
                "success": False,
                "message": f"Không tìm thấy thông tin ban lãnh đạo cho mã {ticker}",
                "data": None
            }, ensure_ascii=False)
        
        # Format data
        formatted_officers = []
        for _, officer in officers.iterrows():
            formatted_officers.append({
                "name": officer.get('officer_name', 'N/A'),
                "position": officer.get('officer_position', 'N/A'),
                "ownership_percent": float(officer.get('officer_own_percent', 0)) * 100,
                "quantity": int(officer.get('quantity', 0)),
                "update_date": str(officer.get('update_date', 'N/A'))
            })
        
        result = {
            "success": True,
            "ticker": ticker,
            "total_officers": len(officers),
            "officers": formatted_officers,
            "message": f"Đã tìm thấy {len(officers)} thành viên ban lãnh đạo của {ticker}"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Lỗi khi lấy thông tin ban lãnh đạo {ticker}: {str(e)}",
            "data": None
        }, ensure_ascii=False)


@tool("get_subsidiaries")
def get_subsidiaries(ticker: str) -> str:
    """
    Lấy danh sách công ty con và công ty liên kết.
    
    Args:
        ticker: Mã chứng khoán viết HOA (VD: VNM, VCB, HPG)
        
    Returns:
        JSON string chứa danh sách công ty con/liên kết
    """
    try:
        # Handle both direct string and wrapped dict arguments
        if isinstance(ticker, dict):
            if 'parameters' in ticker:
                ticker = ticker['parameters'].get('ticker', '')
            elif 'ticker' in ticker:
                ticker = ticker['ticker']
        
        ticker = str(ticker).upper().strip()
        
        if not ticker or len(ticker) < 3:
            return json.dumps({
                "success": False,
                "message": "Mã chứng khoán không hợp lệ."
            }, ensure_ascii=False)
        
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        subsidiaries = stock.company.subsidiaries()
        
        if subsidiaries is None or (isinstance(subsidiaries, pd.DataFrame) and subsidiaries.empty):
            return json.dumps({
                "success": False,
                "message": f"Không tìm thấy thông tin công ty con cho mã {ticker}",
                "data": None
            }, ensure_ascii=False)
        
        # Format data
        formatted_subs = []
        for _, sub in subsidiaries.iterrows():
            formatted_subs.append({
                "name": sub.get('organ_name', 'N/A'),
                "code": sub.get('sub_organ_code', 'N/A'),
                "ownership_percent": float(sub.get('ownership_percent', 0)) * 100,
                "type": sub.get('type', 'N/A')
            })
        
        result = {
            "success": True,
            "ticker": ticker,
            "total_subsidiaries": len(subsidiaries),
            "subsidiaries": formatted_subs,
            "message": f"Đã tìm thấy {len(subsidiaries)} công ty con/liên kết của {ticker}"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Lỗi khi lấy thông tin công ty con {ticker}: {str(e)}",
            "data": None
        }, ensure_ascii=False)


@tool("get_company_events")
def get_company_events(ticker: str) -> str:
    """
    Lấy danh sách sự kiện của công ty (chia cổ tức, họp đại hội cổ đông, v.v.).
    
    Args:
        ticker: Mã chứng khoán viết HOA (VD: VNM, VCB, HPG)
        
    Returns:
        JSON string chứa danh sách sự kiện
    """
    try:
        # Handle both direct string and wrapped dict arguments
        if isinstance(ticker, dict):
            if 'parameters' in ticker:
                ticker = ticker['parameters'].get('ticker', '')
            elif 'ticker' in ticker:
                ticker = ticker['ticker']
        
        ticker = str(ticker).upper().strip()
        
        if not ticker or len(ticker) < 3:
            return json.dumps({
                "success": False,
                "message": "Mã chứng khoán không hợp lệ."
            }, ensure_ascii=False)
        
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        events = stock.company.events()
        
        if events is None or (isinstance(events, pd.DataFrame) and events.empty):
            return json.dumps({
                "success": False,
                "message": f"Không tìm thấy sự kiện cho mã {ticker}",
                "data": None
            }, ensure_ascii=False)
        
        # Get recent 20 events
        recent_events = events.head(20).to_dict('records')
        
        # Format data
        formatted_events = []
        for event in recent_events:
            formatted_events.append({
                "title": event.get('event_title', 'N/A'),
                "type": event.get('event_list_name', 'N/A'),
                "public_date": str(event.get('public_date', 'N/A')),
                "record_date": str(event.get('record_date', 'N/A')),
                "exright_date": str(event.get('exright_date', 'N/A')),
                "ratio": str(event.get('ratio', 'N/A')),
                "value": str(event.get('value', 'N/A'))
            })
        
        result = {
            "success": True,
            "ticker": ticker,
            "total_events": len(events),
            "recent_events": formatted_events,
            "message": f"Đã tìm thấy {len(events)} sự kiện của {ticker}"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Lỗi khi lấy sự kiện {ticker}: {str(e)}",
            "data": None
        }, ensure_ascii=False)


def wrap_tool_result(
    raw_result: str,
    tool_name: str,
    ticker: str,
    llm = None,
    reasoning: str = "Financial data retrieval"
) -> dict:
    """Wrap tool result with enhanced context: data, reasoning, summary.
    
    Args:
        raw_result: Original JSON string result from tool
        tool_name: Name of the tool called
        ticker: Stock ticker
        llm: Language model for summarization (optional)
        reasoning: Why this tool was called
        
    Returns:
        Enhanced dict with {data, reasoning, summary, metrics}
    """
    try:
        from ..core.summarization import create_enhanced_tool_result
        
        # Parse raw result
        result_dict = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        
        # Create enhanced result (skip summary if no LLM provided)
        if llm:
            enhanced = create_enhanced_tool_result(
                data=result_dict,
                tool_name=tool_name,
                llm=llm,
                reasoning=reasoning,
                raw_result=result_dict
            )
        else:
            # Fallback without summarization
            enhanced = {
                "data": result_dict,
                "tool": tool_name,
                "reasoning": reasoning,
                "summary": None,
                "metrics": {}
            }
        return enhanced
    except Exception as e:
        # Fallback: return raw result on error
        return {
            "data": json.loads(raw_result) if isinstance(raw_result, str) else raw_result,
            "tool": tool_name,
            "reasoning": reasoning,
            "summary": None,
            "metrics": {}
        }


def get_vnstock_tools():
    """Get all VnStock tools for the agent"""
    return [
        get_company_info,
        get_historical_data,
        get_shareholders,
        get_officers,
        get_subsidiaries,
        get_company_events
    ]
