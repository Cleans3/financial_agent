"""Response validation for LLM outputs - 3-layer validation."""

import re
import json
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ResponseValidator:
    """
    3-Layer validation for LLM responses.
    Layer 1: Check for explanation keywords (JSON structure talk)
    Layer 2: Check for actual formatted data (tables, structured output)
    Layer 3: Extract and reformat if needed
    """

    EXPLANATION_KEYWORDS = {
        "json_structure": [
            "đối tượng json", "cấu trúc json", "json chứa", "mảng dữ liệu",
            "từng phần tử", "bản ghi dữ liệu", "json parsing", "json.loads"
        ],
        "tool_usage": [
            "sửa đổi mã", "cách sửa", "thêm tham số", "để gọi công cụ",
            "def get_", "import requests", "import json"
        ]
    }

    def validate(self, response: str) -> Tuple[bool, str]:
        """
        Validate response through 3-layer process.

        Returns:
            (is_valid, processed_response)
        """
        # Layer 1: Detect explanations (bad)
        if self._is_explanation(response):
            logger.error("❌ Layer 1 FAILED: Response is explanation, not data")
            # Layer 3: Try to extract data
            extracted = self._extract_json_from_explanation(response)
            if extracted:
                return True, extracted
            return False, response

        # Layer 2: Detect valid data formats (good)
        if self._is_valid_format(response):
            logger.info("✓ Layer 2 PASSED: Response is valid formatted data")
            return True, response

        # Default: Accept as valid
        logger.info("✓ Neutral: Response accepted as-is")
        return True, response

    def _is_explanation(self, response: str) -> bool:
        """Layer 1: Check if response explains JSON/tool usage instead of showing data."""
        response_lower = response.lower()

        for category, keywords in self.EXPLANATION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in response_lower:
                    logger.error(f"  Found {category} keyword: '{keyword}'")
                    return True

        return False

    def _is_valid_format(self, response: str) -> bool:
        """Layer 2: Check if response contains actual formatted data."""
        # Check for Markdown table
        if re.search(r'\|\s*[A-Za-z0-9_\u0080-\uffff\s]+\s*\|', response):
            logger.info("  Found Markdown table marker")
            return True

        # Check for date patterns (YYYY-MM-DD)
        if re.search(r'\d{4}-\d{2}-\d{2}', response):
            logger.info("  Found date pattern (YYYY-MM-DD)")
            return True

        # Check for numeric data with currency/percentage
        if re.search(r'[\d,]+\s*(?:đ|%|VND|USD)', response):
            logger.info("  Found numeric data with currency/percentage")
            return True

        return False

    def _extract_json_from_explanation(self, response: str) -> Optional[str]:
        """Layer 3: Try to extract JSON data from explanation text."""
        logger.info("  Attempting Layer 3: JSON extraction from explanation...")

        json_patterns = [
            r'\{[^{}]*"success"[^{}]*\}',
            r'\[\s*\{[^}]*\}\s*(?:,\s*\{[^}]*\})*\s*\]',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    logger.info("  ✓ Extracted JSON data")
                    return json.dumps(data, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    continue

        logger.warning("  Could not extract JSON from explanation")
        return None
