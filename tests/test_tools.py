"""
Test VnStock Tools - Kiểm tra các công cụ VnStock
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from src.tools import get_company_info, get_historical_data, calculate_sma, calculate_rsi


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def test_company_info():
    """Test getting company information"""
    print_header("TEST 1: Tra cứu thông tin công ty")
    
    tickers = ["VNM", "VCB", "HPG"]
    
    for ticker in tickers:
        print(f"\n--- Testing {ticker} ---")
        result = get_company_info.invoke({"ticker": ticker})
        data = json.loads(result)
        
        if data["success"]:
            print(f"✅ {ticker}: {data['data']['company_name']}")
            print(f"   Sàn: {data['data']['exchange']}")
            print(f"   Ngành: {data['data']['industry']}")
        else:
            print(f"❌ {ticker}: {data['message']}")


def test_historical_data():
    """Test getting historical price data"""
    print_header("TEST 2: Dữ liệu giá lịch sử")
    
    # Test 1: With date range
    print("\n--- Test với khoảng ngày cụ thể (VNM: 2023-01-01 to 2023-06-30) ---")
    result = get_historical_data.invoke({
        "ticker": "VNM",
        "start_date": "2023-01-01",
        "end_date": "2023-06-30"
    })
    data = json.loads(result)
    
    if data["success"]:
        stats = data["statistics"]
        print(f"✅ Tìm thấy {stats['total_records']} bản ghi")
        print(f"   Giá cao nhất: {stats['highest_price']:.2f}")
        print(f"   Giá thấp nhất: {stats['lowest_price']:.2f}")
        print(f"   Biến động: {stats['price_change_percent']:.2f}%")
    else:
        print(f"❌ {data['message']}")
    
    # Test 2: With period
    print("\n--- Test với period (VCB: 3 tháng gần nhất) ---")
    result = get_historical_data.invoke({
        "ticker": "VCB",
        "period": "3M"
    })
    data = json.loads(result)
    
    if data["success"]:
        stats = data["statistics"]
        print(f"✅ Tìm thấy {stats['total_records']} bản ghi")
        print(f"   Giá đóng cửa mới nhất: {stats['latest_close']:.2f}")
        print(f"   Biến động: {stats['price_change_percent']:.2f}%")
    else:
        print(f"❌ {data['message']}")


def test_sma():
    """Test SMA calculation"""
    print_header("TEST 3: Tính SMA (Simple Moving Average)")
    
    print("\n--- Tính SMA-20 cho HPG ---")
    result = calculate_sma.invoke({
        "ticker": "HPG",
        "window": 20,
        "start_date": "2023-01-01",
        "end_date": "2023-06-30"
    })
    data = json.loads(result)
    
    if data["success"]:
        current = data["current_values"]
        analysis = data["analysis"]
        print(f"✅ SMA-20: {current['sma']:.2f}")
        print(f"   Giá hiện tại: {current['price']:.2f}")
        print(f"   Chênh lệch: {current['difference']:.2f} ({current['difference_percent']:.2f}%)")
        print(f"   Xu hướng: {analysis['trend']}")
        print(f"   📊 {analysis['interpretation']}")
    else:
        print(f"❌ {data['message']}")


def test_rsi():
    """Test RSI calculation"""
    print_header("TEST 4: Tính RSI (Relative Strength Index)")
    
    print("\n--- Tính RSI-14 cho VIC ---")
    result = calculate_rsi.invoke({
        "ticker": "VIC",
        "window": 14
    })
    data = json.loads(result)
    
    if data["success"]:
        current = data["current_values"]
        analysis = data["analysis"]
        print(f"✅ RSI-14: {current['rsi']:.2f}")
        print(f"   Giá hiện tại: {current['price']:.2f}")
        print(f"   Trạng thái: {analysis['status']}")
        print(f"   Tín hiệu: {analysis['signal']}")
        print(f"   ⚠️  {analysis['warning']}")
    else:
        print(f"❌ {data['message']}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Testing Financial Agent Tools                       ║
║          VnStock API + Technical Analysis                    ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    try:
        test_company_info()
        test_historical_data()
        test_sma()
        test_rsi()
        
        print("\n" + "="*70)
        print("  ✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
