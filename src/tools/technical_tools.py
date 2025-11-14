"""
Technical Analysis Tools - Tính toán chỉ số kỹ thuật (SMA, RSI)
Sử dụng thư viện ta (technical-analysis)
"""

import json
from typing import Optional
from datetime import datetime, timedelta
from langchain_core.tools import tool
from pydantic import BaseModel, Field, validator
from vnstock import Vnstock
import pandas as pd
import ta
import re


# ============================================
# Pydantic Models cho Tool Input Validation
# ============================================

class SMAInput(BaseModel):
    """Input schema for SMA calculation"""
    ticker: str = Field(
        description="Mã chứng khoán viết HOA (VD: VNM, VCB, HPG)"
    )
    window: int = Field(
        default=20,
        description="Số ngày tính trung bình động (mặc định 20)"
    )
    start_date: Optional[str] = Field(
        None,
        description="Ngày bắt đầu format YYYY-MM-DD"
    )
    end_date: Optional[str] = Field(
        None,
        description="Ngày kết thúc format YYYY-MM-DD"
    )
    
    @validator('ticker')
    def validate_ticker(cls, value):
        return value.upper().strip()
    
    @validator('window')
    def validate_window(cls, value):
        if value < 1 or value > 200:
            raise ValueError("Window size phải từ 1 đến 200 ngày")
        return value
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, value):
        if value is None:
            return value
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, value):
            raise ValueError("Ngày phải có format YYYY-MM-DD")
        try:
            datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Ngày không hợp lệ: {value}")
        return value


class RSIInput(BaseModel):
    """Input schema for RSI calculation"""
    ticker: str = Field(
        description="Mã chứng khoán viết HOA (VD: VNM, VCB, HPG)"
    )
    window: int = Field(
        default=14,
        description="Số ngày tính RSI (mặc định 14)"
    )
    start_date: Optional[str] = Field(
        None,
        description="Ngày bắt đầu format YYYY-MM-DD"
    )
    end_date: Optional[str] = Field(
        None,
        description="Ngày kết thúc format YYYY-MM-DD"
    )
    
    @validator('ticker')
    def validate_ticker(cls, value):
        return value.upper().strip()
    
    @validator('window')
    def validate_window(cls, value):
        if value < 2 or value > 100:
            raise ValueError("Window size cho RSI phải từ 2 đến 100 ngày")
        return value
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, value):
        if value is None:
            return value
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, value):
            raise ValueError("Ngày phải có format YYYY-MM-DD")
        try:
            datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Ngày không hợp lệ: {value}")
        return value


# ============================================
# Tool Implementations
# ============================================

@tool("calculate_sma", args_schema=SMAInput)
def calculate_sma(
    ticker: str,
    window: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    Tính Simple Moving Average (SMA) cho mã chứng khoán.
    
    SMA là trung bình động đơn giản, giúp làm mượt biến động giá và xác định xu hướng.
    - SMA tăng dần: Xu hướng tăng
    - SMA giảm dần: Xu hướng giảm
    - Giá > SMA: Tín hiệu tích cực
    - Giá < SMA: Tín hiệu tiêu cực
    
    Args:
        ticker: Mã chứng khoán viết HOA (VD: VNM, VCB, HPG)
        window: Số ngày tính trung bình (mặc định 20)
        start_date: Ngày bắt đầu format YYYY-MM-DD
        end_date: Ngày kết thúc format YYYY-MM-DD
        
    Returns:
        JSON string chứa kết quả tính SMA và phân tích xu hướng
        
    Example:
        calculate_sma(ticker="HPG", window=20, start_date="2023-01-01", end_date="2023-06-30")
    """
    try:
        ticker = ticker.upper().strip()
        
        # Default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Need extra days for SMA calculation
            start_date = (datetime.now() - timedelta(days=window * 2 + 90)).strftime("%Y-%m-%d")
        
        # Get historical data
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        df = stock.quote.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            return json.dumps({
                "success": False,
                "message": f"Không có dữ liệu để tính SMA cho {ticker}",
                "data": None
            }, ensure_ascii=False)
        
        # Calculate SMA
        df[f'SMA_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
        
        # Filter out NaN rows
        df_clean = df.dropna()
        
        if df_clean.empty:
            return json.dumps({
                "success": False,
                "message": f"Không đủ dữ liệu để tính SMA-{window} cho {ticker}",
                "data": None
            }, ensure_ascii=False)
        
        # Get recent data (last 20 records)
        recent_data = df_clean.tail(20)
        
        # Current values
        current_price = float(df_clean['close'].iloc[-1])
        current_sma = float(df_clean[f'SMA_{window}'].iloc[-1])
        
        # Trend analysis
        if current_price > current_sma:
            trend = "TĂNG - Giá đang trên SMA (tín hiệu tích cực)"
            signal = "positive"
        else:
            trend = "GIẢM - Giá đang dưới SMA (tín hiệu tiêu cực)"
            signal = "negative"
        
        # SMA slope (increasing or decreasing)
        sma_change = float(current_sma - df_clean[f'SMA_{window}'].iloc[-5])
        sma_trend = "tăng" if sma_change > 0 else "giảm"
        
        # Format recent data for output
        recent_records = []
        for idx, row in recent_data.iterrows():
            recent_records.append({
                "date": row['time'].strftime('%Y-%m-%d') if isinstance(row['time'], pd.Timestamp) else str(row['time']),
                "close": float(row['close']),
                f"sma_{window}": float(row[f'SMA_{window}']),
                "difference": float(row['close'] - row[f'SMA_{window}']),
                "difference_percent": float(((row['close'] - row[f'SMA_{window}']) / row[f'SMA_{window}']) * 100)
            })
        
        result = {
            "success": True,
            "ticker": ticker,
            "indicator": f"SMA-{window}",
            "current_values": {
                "price": current_price,
                "sma": current_sma,
                "difference": current_price - current_sma,
                "difference_percent": ((current_price - current_sma) / current_sma) * 100
            },
            "analysis": {
                "trend": trend,
                "signal": signal,
                "sma_trend": sma_trend,
                "interpretation": f"Giá hiện tại {'cao hơn' if current_price > current_sma else 'thấp hơn'} SMA-{window} khoảng {abs(current_price - current_sma):.2f} điểm ({abs((current_price - current_sma) / current_sma * 100):.2f}%). SMA đang {sma_trend}."
            },
            "detailed_data": recent_records,  # Full detailed data for table display
            "message": f"Đã tính SMA-{window} cho {ticker} thành công. Dữ liệu chi tiết gồm {len(recent_records)} ngày gần nhất."
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Lỗi khi tính SMA cho {ticker}: {str(e)}",
            "data": None
        }, ensure_ascii=False)


@tool("calculate_rsi", args_schema=RSIInput)
def calculate_rsi(
    ticker: str,
    window: int = 14,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    Tính Relative Strength Index (RSI) cho mã chứng khoán.
    
    RSI là chỉ số dao động từ 0-100, đo lường động lượng giá:
    - RSI > 70: QUÁ MUA (Overbought) - Giá có thể điều chỉnh giảm
    - RSI < 30: QUÁ BÁN (Oversold) - Giá có thể phục hồi tăng
    - 30 < RSI < 70: TRUNG TÍNH - Giá đang trong vùng cân bằng
    
    Args:
        ticker: Mã chứng khoán viết HOA (VD: VNM, VCB, HPG)
        window: Số ngày tính RSI (mặc định 14)
        start_date: Ngày bắt đầu format YYYY-MM-DD
        end_date: Ngày kết thúc format YYYY-MM-DD
        
    Returns:
        JSON string chứa kết quả tính RSI và phân tích
        
    Example:
        calculate_rsi(ticker="VIC", window=14, start_date="2023-05-01", end_date="2023-05-31")
    """
    try:
        ticker = ticker.upper().strip()
        
        # Default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Need extra days for RSI calculation
            start_date = (datetime.now() - timedelta(days=window * 2 + 90)).strftime("%Y-%m-%d")
        
        # Get historical data
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        df = stock.quote.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            return json.dumps({
                "success": False,
                "message": f"Không có dữ liệu để tính RSI cho {ticker}",
                "data": None
            }, ensure_ascii=False)
        
        # Calculate RSI
        df[f'RSI_{window}'] = ta.momentum.rsi(df['close'], window=window)
        
        # Filter out NaN rows
        df_clean = df.dropna()
        
        if df_clean.empty:
            return json.dumps({
                "success": False,
                "message": f"Không đủ dữ liệu để tính RSI-{window} cho {ticker}",
                "data": None
            }, ensure_ascii=False)
        
        # Get recent data
        recent_data = df_clean.tail(20)
        
        # Current values
        current_rsi = float(df_clean[f'RSI_{window}'].iloc[-1])
        current_price = float(df_clean['close'].iloc[-1])
        
        # RSI analysis
        if current_rsi > 70:
            status = "QUÁ MUA (Overbought)"
            signal = "sell"
            warning = "Cảnh báo: Giá có thể điều chỉnh giảm trong ngắn hạn"
            color = "red"
        elif current_rsi < 30:
            status = "QUÁ BÁN (Oversold)"
            signal = "buy"
            warning = "Cơ hội: Giá có thể phục hồi tăng trong ngắn hạn"
            color = "green"
        else:
            status = "TRUNG TÍNH"
            signal = "neutral"
            warning = "Giá đang trong vùng cân bằng, chưa có tín hiệu rõ ràng"
            color = "yellow"
        
        # Format recent data
        recent_records = []
        for idx, row in recent_data.iterrows():
            rsi_value = float(row[f'RSI_{window}'])
            # Determine status for each row
            if rsi_value > 70:
                row_status = "Quá mua"
            elif rsi_value < 30:
                row_status = "Quá bán"
            else:
                row_status = "Trung tính"
                
            recent_records.append({
                "date": row['time'].strftime('%Y-%m-%d') if isinstance(row['time'], pd.Timestamp) else str(row['time']),
                "close": float(row['close']),
                f"rsi_{window}": rsi_value,
                "status": row_status
            })
        
        result = {
            "success": True,
            "ticker": ticker,
            "indicator": f"RSI-{window}",
            "current_values": {
                "price": current_price,
                "rsi": current_rsi
            },
            "analysis": {
                "status": status,
                "signal": signal,
                "warning": warning,
                "color": color,
                "interpretation": f"RSI hiện tại là {current_rsi:.2f}. {status}. {warning}"
            },
            "thresholds": {
                "overbought": 70,
                "oversold": 30,
                "neutral_range": "30-70"
            },
            "detailed_data": recent_records,  # Full detailed data for table display
            "message": f"Đã tính RSI-{window} cho {ticker} thành công. Dữ liệu chi tiết gồm {len(recent_records)} ngày gần nhất."
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Lỗi khi tính RSI cho {ticker}: {str(e)}",
            "data": None
        }, ensure_ascii=False)


def get_technical_tools():
    """Get all technical analysis tools"""
    return [calculate_sma, calculate_rsi]
