"""
Example Excel file generator for test questions
Creates a sample Excel file with test questions
"""

import pandas as pd
from datetime import datetime


def create_sample_test_file(filename='test_questions_sample.xlsx'):
    """Create sample Excel file with test questions"""
    
    # Sample test questions
    test_data = [
        {
            'STT': 1,
            'Câu hỏi': 'Thông tin công ty VNM',
            'Loại câu hỏi': 'Thông tin doanh nghiệp',
            'Câu trả lời mong đợi': 'Thông tin về Vinamilk (mã VNM)',
            'Ghi chú': 'Test tool get_company_info'
        },
        {
            'STT': 2,
            'Câu hỏi': 'Cổ đông lớn của VCB là ai?',
            'Loại câu hỏi': 'Cổ đông',
            'Câu trả lời mong đợi': 'Danh sách top cổ đông VCB',
            'Ghi chú': 'Test tool get_shareholders'
        },
        {
            'STT': 3,
            'Câu hỏi': 'Ban lãnh đạo của HPG gồm những ai?',
            'Loại câu hỏi': 'Ban lãnh đạo',
            'Câu trả lời mong đợi': 'Danh sách lãnh đạo HPG',
            'Ghi chú': 'Test tool get_officers'
        },
        {
            'STT': 4,
            'Câu hỏi': 'VNM có công ty con nào?',
            'Loại câu hỏi': 'Công ty con',
            'Câu trả lời mong đợi': 'Danh sách công ty con của VNM',
            'Ghi chú': 'Test tool get_subsidiaries'
        },
        {
            'STT': 5,
            'Câu hỏi': 'Giá cổ phiếu VCB trong 3 tháng gần nhất',
            'Loại câu hỏi': 'Dữ liệu lịch sử',
            'Câu trả lời mong đợi': 'OHLCV của VCB 3 tháng',
            'Ghi chú': 'Test tool get_historical_data'
        },
        {
            'STT': 6,
            'Câu hỏi': 'Tính SMA 20 ngày cho HPG',
            'Loại câu hỏi': 'Phân tích kỹ thuật',
            'Câu trả lời mong đợi': 'SMA-20 của HPG với xu hướng',
            'Ghi chú': 'Test tool calculate_sma'
        },
        {
            'STT': 7,
            'Câu hỏi': 'RSI của VIC hiện tại là bao nhiêu?',
            'Loại câu hỏi': 'Phân tích kỹ thuật',
            'Câu trả lời mong đợi': 'RSI của VIC với đánh giá',
            'Ghi chú': 'Test tool calculate_rsi'
        },
        {
            'STT': 8,
            'Câu hỏi': 'Sự kiện gần đây của FPT',
            'Loại câu hỏi': 'Sự kiện doanh nghiệp',
            'Câu trả lời mong đợi': 'Các sự kiện gần đây của FPT',
            'Ghi chú': 'Test tool get_company_events'
        },
        {
            'STT': 9,
            'Câu hỏi': 'So sánh giá VNM và VCB trong 1 tháng',
            'Loại câu hỏi': 'Phân tích đa mã',
            'Câu trả lời mong đợi': 'Dữ liệu giá của cả VNM và VCB',
            'Ghi chú': 'Test nhiều tool'
        },
        {
            'STT': 10,
            'Câu hỏi': 'HPG có quá mua hay quá bán không?',
            'Loại câu hỏi': 'Phân tích kỹ thuật',
            'Câu trả lời mong đợi': 'RSI và đánh giá trạng thái',
            'Ghi chú': 'Test RSI và interpretation'
        },
        {
            'STT': 11,
            'Câu hỏi': 'Xu hướng giá VIC trong 6 tháng',
            'Loại câu hỏi': 'Phân tích xu hướng',
            'Câu trả lời mong đợi': 'Dữ liệu giá và SMA',
            'Ghi chú': 'Test historical data và SMA'
        },
        {
            'STT': 12,
            'Câu hỏi': 'Thông tin ngành của MBB',
            'Loại câu hỏi': 'Thông tin doanh nghiệp',
            'Câu trả lời mong đợi': 'Thông tin ngành và công ty MBB',
            'Ghi chú': 'Test company info với focus vào ngành'
        },
    ]
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    # Save to Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Test Questions', index=False)
        
        # Add instructions sheet
        instructions = pd.DataFrame({
            'Hướng dẫn': [
                '1. File này chứa các câu hỏi test cho Financial Agent',
                '2. Cột "Câu hỏi" chứa câu hỏi cần test',
                '3. Cột "Loại câu hỏi" phân loại câu hỏi',
                '4. Cột "Câu trả lời mong đợi" mô tả kết quả mong đợi',
                '5. Chạy test bằng lệnh: python test_auto.py test_questions_sample.xlsx',
                '6. Kết quả sẽ được lưu trong file test_results_[timestamp].xlsx',
                '',
                'Các loại câu hỏi:',
                '- Thông tin doanh nghiệp: get_company_info',
                '- Cổ đông: get_shareholders',
                '- Ban lãnh đạo: get_officers',
                '- Công ty con: get_subsidiaries',
                '- Sự kiện doanh nghiệp: get_company_events',
                '- Dữ liệu lịch sử: get_historical_data',
                '- Phân tích kỹ thuật: calculate_sma, calculate_rsi',
            ]
        })
        instructions.to_excel(writer, sheet_name='Instructions', index=False)
    
    print(f"✅ Created sample test file: {filename}")
    return filename


if __name__ == "__main__":
    create_sample_test_file()
