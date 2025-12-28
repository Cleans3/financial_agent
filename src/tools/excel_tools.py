"""
Excel Analysis Tools - Chuyển đổi file Excel thành Markdown cho Gemini phân tích tài chính
"""

import pandas as pd
import logging
import os
import json
from pathlib import Path
from typing import Optional, Dict, List
from langchain_core.tools import tool

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_excel_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load file Excel và trích xuất tất cả các sheet
    
    Args:
        file_path: Đường dẫn tới file Excel
        
    Returns:
        Dictionary với tên sheet và DataFrame
    """
    try:
        logger.info(f"Loading Excel file: {file_path}")
        excel_file = pd.ExcelFile(file_path)
        
        sheets_dict = {}
        for sheet_name in excel_file.sheet_names:
            logger.info(f"Reading sheet: {sheet_name}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            sheets_dict[sheet_name] = df
            
        logger.info(f"✓ Loaded {len(sheets_dict)} sheets")
        return sheets_dict
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise


def detect_and_split_tables(df: pd.DataFrame, sheet_name: str = "") -> List[tuple]:
    """
    Detect multiple tables within a single sheet by finding empty row separators.
    
    A new table is detected when:
    - There are 1+ consecutive completely empty rows (separator)
    - Column structure changes significantly
    - A header-like row appears (after empty rows)
    
    Args:
        df: DataFrame from a sheet
        sheet_name: Name of the sheet (for naming tables)
        
    Returns:
        List of (table_df, table_name) tuples
    """
    try:
        logger.info(f"Detecting tables in sheet: {sheet_name}")
        
        if df.empty:
            return []
        
        # Get indices of completely empty rows
        empty_row_indices = []
        for idx, row in df.iterrows():
            if row.isna().all():
                empty_row_indices.append(idx)
        
        # If no empty rows, treat entire sheet as single table
        if not empty_row_indices:
            logger.info(f"No empty rows detected - treating sheet as single table")
            return [(df, sheet_name)]
        
        # Find groups of consecutive empty rows (each group = separator)
        table_separators = []
        i = 0
        while i < len(empty_row_indices):
            group_start = empty_row_indices[i]
            group_end = empty_row_indices[i]
            
            # Find consecutive empty rows
            while i + 1 < len(empty_row_indices) and empty_row_indices[i + 1] == empty_row_indices[i] + 1:
                i += 1
                group_end = empty_row_indices[i]
            
            # Mark the end of the empty row group as separator
            table_separators.append(group_end)
            i += 1
        
        if not table_separators:
            logger.info(f"No empty row separators found - treating as single table")
            return [(df, sheet_name)]
        
        # Split dataframe by separators
        tables = []
        start_idx = 0
        
        for sep_idx in table_separators:
            if sep_idx > start_idx:
                # Extract table section (before the separator)
                table_section = df.iloc[start_idx:sep_idx].copy()
                table_section = clean_dataframe(table_section)
                
                if not table_section.empty:
                    # Auto-generate table name
                    table_num = len(tables) + 1
                    table_name = f"{sheet_name} - Bảng {table_num}" if sheet_name else f"Bảng {table_num}"
                    
                    # Try to extract title from first row if it looks like a header
                    first_row = table_section.iloc[0]
                    if not any(pd.isna(first_row)):
                        # First row has content - might be a title
                        first_row_str = str(first_row.iloc[0])
                        if len(first_row_str) > 5 and len(first_row_str) < 100:
                            table_name = f"{sheet_name} - {first_row_str}" if sheet_name else first_row_str
                            table_section = table_section.iloc[1:].reset_index(drop=True)
                    
                    tables.append((table_section, table_name))
                    logger.info(f"  Detected table: {table_name} ({len(table_section)} rows)")
            
            # Skip past the separator (empty rows)
            start_idx = sep_idx + 1
        
        # Add remaining data after last separator
        if start_idx < len(df):
            remaining = df.iloc[start_idx:].copy()
            remaining = clean_dataframe(remaining)
            if not remaining.empty:
                table_num = len(tables) + 1
                table_name = f"{sheet_name} - Bảng {table_num}" if sheet_name else f"Bảng {table_num}"
                tables.append((remaining, table_name))
                logger.info(f"  Detected table: {table_name} ({len(remaining)} rows)")
        
        logger.info(f"✓ Detected {len(tables)} tables in sheet")
        return tables if tables else [(df, sheet_name)]
        
    except Exception as e:
        logger.error(f"Error detecting tables: {e}")
        return [(df, sheet_name)]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch DataFrame - loại bỏ hàng/cột trống, chuẩn hóa dữ liệu
    
    Args:
        df: DataFrame cần làm sạch
        
    Returns:
        DataFrame đã làm sạch
    """
    try:
        logger.info(f"Cleaning DataFrame with shape {df.shape}")
        
        # Loại bỏ hàng hoàn toàn trống
        df = df.dropna(how='all')
        
        # Loại bỏ cột hoàn toàn trống
        df = df.dropna(axis=1, how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info(f"✓ Cleaned to shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error cleaning DataFrame: {e}")
        raise


def format_number(value) -> str:
    """
    Định dạng số với dấu phẩy tách hàng nghìn
    
    Args:
        value: Giá trị cần định dạng
        
    Returns:
        Chuỗi số đã định dạng
    """
    try:
        # Xử lý các loại dữ liệu khác nhau
        if pd.isna(value) or value == '':
            return ''
        
        if isinstance(value, str):
            return value.strip()
        
        # Chuyển đổi sang float
        if isinstance(value, (int, float)):
            # Nếu là số nguyên lớn hơn 1000, thêm dấu phẩy
            if isinstance(value, float) and value == int(value):
                value = int(value)
            
            if isinstance(value, int) and abs(value) >= 1000:
                return f"{value:,}".replace(',', '.')  # Sử dụng dấu chấm cho hàng nghìn (tùy chọn)
            elif isinstance(value, int):
                return str(value)
            else:
                # Làm tròn 2 chữ số sau dấu phẩy cho số thập phân
                return f"{value:.2f}"
        
        return str(value)
    except Exception as e:
        logger.warning(f"Error formatting number {value}: {e}")
        return str(value)


def format_number_vn(value) -> str:
    """
    Định dạng số theo tiêu chuẩn Việt Nam (dấu phẩy tách hàng nghìn)
    
    Args:
        value: Giá trị cần định dạng
        
    Returns:
        Chuỗi số đã định dạng (VN style)
    """
    try:
        if pd.isna(value) or value == '':
            return ''
        
        if isinstance(value, str):
            return value.strip()
        
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value == int(value):
                value = int(value)
            
            if isinstance(value, int) and abs(value) >= 1000:
                # Định dạng: 1,234,567 (dùng dấu phẩy)
                return f"{value:,}"
            elif isinstance(value, int):
                return str(value)
            else:
                return f"{value:.2f}"
        
        return str(value)
    except:
        return str(value)


def dataframe_to_markdown(df: pd.DataFrame, title: str = "", unit: str = "", max_rows: int = 200) -> str:
    """
    Chuyển đổi DataFrame thành bảng Markdown (với giới hạn hàng để tránh vượt token limit)
    
    Args:
        df: DataFrame cần chuyển đổi
        title: Tiêu đề bảng (VD: "Cân đối kế toán")
        unit: Đơn vị dữ liệu (VD: "VNĐ", "Năm", "Tháng")
        max_rows: Số hàng tối đa hiển thị (mặc định 100 để tránh vượt token limit Gemini)
        
    Returns:
        Chuỗi Markdown định dạng bảng
    """
    try:
        df = clean_dataframe(df)
        
        if df.empty:
            logger.warning("DataFrame is empty, skipping")
            return ""
        
        # Giới hạn số hàng nếu vượt quá max_rows
        total_rows = len(df)
        is_truncated = total_rows > max_rows
        if is_truncated:
            logger.info(f"Truncating DataFrame from {total_rows} to {max_rows} rows")
            df_display = df.head(max_rows)
        else:
            df_display = df
        
        # Định dạng các cột số
        df_formatted = df_display.copy()
        for col in df_formatted.columns:
            if df_display[col].dtype in ['int64', 'float64']:
                df_formatted[col] = df_formatted[col].apply(format_number_vn)
        
        # Tạo tiêu đề
        markdown = ""
        if title:
            if unit:
                markdown += f"**{title} ({unit})**"
            else:
                markdown += f"**{title}**"
            
            # Thêm info về số hàng
            if is_truncated:
                markdown += f" — Hiển thị {max_rows}/{total_rows} hàng"
            else:
                markdown += f" — {total_rows} hàng"
            markdown += "\n\n"
        
        # Tạo header
        markdown += "| " + " | ".join(df_formatted.columns.astype(str)) + " |\n"
        
        # Tạo separator
        markdown += "| " + " | ".join(["---"] * len(df_formatted.columns)) + " |\n"
        
        # Tạo dữ liệu
        for _, row in df_formatted.iterrows():
            markdown += "| " + " | ".join(row.astype(str)) + " |\n"
        
        # Thêm ghi chú nếu bị cắt
        if is_truncated:
            markdown += f"\n*⚠️ Bảng này hiển thị {max_rows} hàng đầu tiên trong tổng số {total_rows} hàng. Vui lòng tham khảo file Excel gốc để xem toàn bộ dữ liệu.*\n\n"
        else:
            markdown += "\n"
        
        logger.info(f"✓ Converted DataFrame to Markdown ({len(df_display)} rows, {len(df_formatted.columns)} cols)")
        return markdown
    except Exception as e:
        logger.error(f"Error converting DataFrame to Markdown: {e}")
        return ""


def analyze_excel_to_markdown(file_path: str, max_rows_per_sheet: int = 200) -> Dict:
    """
    Phân tích file Excel và chuyển đổi thành Markdown cho Gemini
    (Với giới hạn hàng mỗi sheet để tránh vượt token limit)
    
    Args:
        file_path: Đường dẫn tới file Excel
        max_rows_per_sheet: Số hàng tối đa mỗi sheet (mặc định 100)
        
    Returns:
        Dictionary với markdown và metadata
    """
    try:
        logger.info(f"Starting Excel analysis: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load file Excel
        sheets_dict = load_excel_file(file_path)
        
        # Xác định loại dữ liệu
        file_name = Path(file_path).stem
        
        # Chuyển đổi tất cả sheet thành Markdown
        markdown_output = f"# Phân tích dữ liệu từ file: {file_name}\n\n"
        
        sheet_names = list(sheets_dict.keys())
        markdown_output += f"**Tóm tắt:** File chứa {len(sheet_names)} bảng tính\n\n"
        markdown_output += f"**Các sheet:** {', '.join(sheet_names)}\n\n"
        markdown_output += "---\n\n"
        
        # Xử lý từng sheet
        for sheet_name, df in sheets_dict.items():
            if df.empty:
                logger.warning(f"Sheet '{sheet_name}' is empty, skipping")
                continue
            
            # DETECT AND SPLIT multiple tables within the sheet
            tables_in_sheet = detect_and_split_tables(df, sheet_name)
            
            logger.info(f"Processing {len(tables_in_sheet)} table(s) from sheet '{sheet_name}'")
            
            for table_df, table_name in tables_in_sheet:
                if table_df.empty:
                    continue
                
                # Làm sạch dữ liệu
                table_df = clean_dataframe(table_df)
                
                # Xác định đơn vị dựa trên tên sheet/table
                unit = "VNĐ"  # Mặc định
                if "%" in sheet_name or "tỷ" in sheet_name.lower():
                    unit = "%"
                elif "cổ phần" in sheet_name.lower() or "số lượng" in sheet_name.lower():
                    unit = "Cổ phần"
                elif "giá" in sheet_name.lower():
                    unit = "VNĐ"
                
                # Chuyển đổi sang Markdown với giới hạn hàng
                markdown_table = dataframe_to_markdown(table_df, title=table_name, unit=unit, max_rows=max_rows_per_sheet)
                markdown_output += markdown_table
        
        logger.info(f"✓ Analysis completed")
        
        return {
            "success": True,
            "file_name": file_name,
            "sheet_count": len(sheets_dict),
            "sheet_names": sheet_names,
            "markdown": markdown_output,
            "message": f"✓ Chuyển đổi thành công {len(sheets_dict)} sheet thành Markdown (tối đa {max_rows_per_sheet} hàng/sheet)"
        }
    except Exception as e:
        logger.error(f"Error analyzing Excel file: {e}")
        return {
            "success": False,
            "file_name": Path(file_path).stem if file_path else "Unknown",
            "sheet_count": 0,
            "sheet_names": [],
            "markdown": "",
            "message": f"❌ Lỗi: {str(e)}"
        }


def get_excel_tools():
    """
    Export các tool xử lý Excel cho LangGraph
    
    Returns:
        List các tool definition
    """
    
    @tool
    def analyze_excel_file(file_path: str) -> str:
        """
        Phân tích file Excel và chuyển đổi thành bảng Markdown cho Gemini AI
        
        Args:
            file_path: Đường dẫn tới file Excel (.xlsx, .xls)
            
        Returns:
            Markdown formatted tables cho Gemini phân tích tài chính
            
        Example:
            >>> analyze_excel_file("/path/to/financial_data.xlsx")
            # Phân tích dữ liệu từ file: financial_data
            
            **Tóm tắt:** File chứa 3 bảng tính
            ...
        """
        try:
            result = analyze_excel_to_markdown(file_path)
            
            if result['success']:
                return result['markdown']
            else:
                return f"❌ Lỗi xử lý file Excel: {result['message']}"
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"
    
    return [analyze_excel_file]
