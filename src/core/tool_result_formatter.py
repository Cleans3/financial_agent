"""Format tool execution results as readable Markdown."""

import logging
from typing import Dict, Union, List, Any

logger = logging.getLogger(__name__)


class ToolResultFormatter:
    """Format tool execution results based on data type."""

    def format(self, data: Union[Dict[str, Any], str, List]) -> str:
        """
        Format tool result based on data type.

        Args:
            data: Tool execution result

        Returns:
            Formatted string (Markdown table or readable format)
        """
        if isinstance(data, str):
            return data

        if not isinstance(data, dict):
            return str(data)

        # RSI result
        if "indicator" in data and data.get("indicator", "").startswith("RSI"):
            return self._format_rsi(data)

        # SMA result
        if "indicator" in data and data.get("indicator", "").startswith("SMA"):
            return self._format_sma(data)

        # Historical data
        if "detailed_data" in data and isinstance(data.get("detailed_data"), list):
            return self._format_historical_data(data)

        # Company info
        if "company" in data or "symbol" in data:
            return self._format_company_info(data)

        # Fallback
        return self._format_generic(data)

    def _format_rsi(self, data: Dict[str, Any]) -> str:
        """Format RSI indicator result as table."""
        logger.info("Formatting RSI result as table")
        rows = ["|Ngày|Giá đóng cửa|RSI|Trạng thái|"]
        rows.append("|---|---|---|---|")

        for item in data.get("detailed_data", [])[:10]:
            date = item.get("date", "")
            close = item.get("close", "")
            rsi = item.get("rsi_14", "")
            status = item.get("status", "")
            rows.append(f"|{date}|{close}|{rsi}|{status}|")

        # Add analysis summary
        if "analysis" in data:
            analysis = data["analysis"]
            status = analysis.get("status", "")
            if status == "OVERBOUGHT":
                rows.append("\n**Nhận xét**: Cổ phiếu đang ở vùng quá mua (RSI > 70)")
            elif status == "OVERSOLD":
                rows.append("\n**Nhận xét**: Cổ phiếu đang ở vùng quá bán (RSI < 30)")

        return "\n".join(rows)

    def _format_sma(self, data: Dict[str, Any]) -> str:
        """Format SMA indicator result as table."""
        logger.info("Formatting SMA result as table")
        rows = ["|Ngày|Giá đóng cửa|SMA20|SMA50|Tín hiệu|"]
        rows.append("|---|---|---|---|---|")

        for item in data.get("detailed_data", [])[:10]:
            date = item.get("date", "")
            close = item.get("close", "")
            sma20 = item.get("sma20", "")
            sma50 = item.get("sma50", "")
            signal = item.get("signal", "")
            rows.append(f"|{date}|{close}|{sma20}|{sma50}|{signal}|")

        if "trend" in data:
            rows.append(f"\n**Xu hướng**: {data['trend']}")

        return "\n".join(rows)

    def _format_historical_data(self, data: Dict[str, Any]) -> str:
        """Format historical price data as table."""
        logger.info("Formatting historical data as table")
        rows = ["|Ngày|Mở|Cao|Thấp|Đóng|Khối lượng|"]
        rows.append("|---|---|---|---|---|---|")

        for item in data.get("detailed_data", [])[:15]:
            date = item.get("date", "")
            open_p = item.get("open", "")
            high = item.get("high", "")
            low = item.get("low", "")
            close = item.get("close", "")
            volume = item.get("volume", "")
            rows.append(f"|{date}|{open_p}|{high}|{low}|{close}|{volume}|")

        return "\n".join(rows)

    def _format_company_info(self, data: Dict[str, Any]) -> str:
        """Format company information."""
        logger.info("Formatting company info")
        rows = []

        if "company" in data:
            rows.append(f"**Công ty**: {data['company']}")
        if "symbol" in data:
            rows.append(f"**Mã chứng chỉ**: {data['symbol']}")
        if "price" in data:
            rows.append(f"**Giá hiện tại**: {data['price']}")
        if "market_cap" in data:
            rows.append(f"**Vốn hóa**: {data['market_cap']}")
        if "pe_ratio" in data:
            rows.append(f"**P/E Ratio**: {data['pe_ratio']}")

        return "\n".join(rows) if rows else str(data)

    def _format_generic(self, data: Dict[str, Any]) -> str:
        """Format generic dictionary as readable text."""
        logger.info("Formatting generic data")
        rows = []

        for key, value in data.items():
            if isinstance(value, (list, dict)):
                rows.append(f"**{key}**: {str(value)[:100]}...")
            else:
                rows.append(f"**{key}**: {value}")

        return "\n".join(rows) if rows else str(data)
