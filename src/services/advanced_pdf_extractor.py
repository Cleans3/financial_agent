"""
Advanced PDF Extraction with Improved Table Detection and Text/Table Separation
Features:
- Stream flavor for table detection (better for text-formatted financial tables)
- Automatic preamble row removal from tables
- Section header cleanup
- Text filtering to remove table content duplication
- Proper document order preservation (content sorted by position)
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import pymupdf

from .pdf_content_detector import PDFContentDetector
from .table_extractor import TableExtractor


class AdvancedPDFExtractor:
    """Extract PDF content in proper document order with table/text separation"""
    
    @staticmethod
    def extract_with_order(pdf_path: str) -> Dict[str, Any]:
        """
        Extract PDF content with proper document order and table/text separation
        
        Returns:
            {
                'success': bool,
                'file_name': str,
                'total_pages': int,
                'content_items': [
                    {
                        'type': 'text' | 'table',
                        'page': int,
                        'text': str (if type='text'),
                        'table': List[List] (if type='table'),
                        'table_index': int (if type='table'),
                        'source': str (if type='table')
                    }
                ],
                'message': str
            }
        """
        try:
            if not Path(pdf_path).exists():
                return {
                    'success': False,
                    'file_name': Path(pdf_path).name if pdf_path else 'unknown',
                    'total_pages': 0,
                    'content_items': [],
                    'message': f'File not found: {pdf_path}'
                }
            
            # Step 1: Extract tables first (to build filter set)
            tables = TableExtractor.extract_tables(pdf_path, use_camelot=True)
            tables_by_page = {}
            for table_info in tables:
                page = table_info['page'] - 1  # Convert to 0-based
                if page not in tables_by_page:
                    tables_by_page[page] = []
                tables_by_page[page].append(table_info)
            
            # Step 2: Build set of all table content for filtering
            table_content_set = set()
            for page_tables in tables_by_page.values():
                for table_info in page_tables:
                    for row in table_info['data']:
                        for cell in row:
                            if cell:
                                cell_str = str(cell).strip()
                                if cell_str:
                                    table_content_set.add(cell_str)
            
            # Step 3: Classify pages
            detector = PDFContentDetector(debug=False)
            classifications = detector.analyze_document(pdf_path)
            
            # Step 4: Extract text blocks with position info
            doc = pymupdf.open(pdf_path)
            total_pages = len(doc)
            
            all_content_items = []
            
            for page_num in range(total_pages):
                page = doc[page_num]
                
                # Get text blocks with position info
                blocks = page.get_text("blocks")
                
                # Build ordered content list for this page
                content_items = []
                
                # Extract text blocks with Y-coordinate
                for block in blocks:
                    x0, y0, x1, y1, text, block_num, block_type = block
                    text_stripped = text.strip()
                    
                    if text_stripped and block_type == 0:  # 0 = text block
                        content_items.append({
                            'type': 'text',
                            'y': y0,
                            'text': text_stripped
                        })
                
                # Add tables with their Y-coordinate
                if page_num in tables_by_page:
                    for idx, table_info in enumerate(tables_by_page[page_num]):
                        y_pos = len(content_items) * 15  # Rough approximation
                        content_items.append({
                            'type': 'table',
                            'y': y_pos,
                            'index': idx,
                            'table_info': table_info
                        })
                
                # Sort by Y coordinate (top to bottom)
                content_items.sort(key=lambda x: x['y'])
                
                # Filter and output in order
                text_buffer = []
                table_count = 0
                
                for item in content_items:
                    if item['type'] == 'text':
                        line = item['text']
                        
                        # Check if this is table content
                        is_table_content = AdvancedPDFExtractor._is_table_content(
                            line, table_content_set
                        )
                        
                        if not is_table_content:
                            text_buffer.append(line)
                    
                    elif item['type'] == 'table':
                        # Output buffered text first
                        if text_buffer:
                            all_content_items.append({
                                'type': 'text',
                                'page': page_num + 1,
                                'text': '\n'.join(text_buffer)
                            })
                            text_buffer = []
                        
                        # Output table
                        table_info = item['table_info']
                        table = table_info['data']
                        table_count += 1
                        
                        all_content_items.append({
                            'type': 'table',
                            'page': page_num + 1,
                            'table_index': table_count,
                            'table': table,
                            'source': table_info['source']
                        })
                
                # Output any remaining text
                if text_buffer:
                    all_content_items.append({
                        'type': 'text',
                        'page': page_num + 1,
                        'text': '\n'.join(text_buffer)
                    })
            
            doc.close()
            
            return {
                'success': True,
                'file_name': Path(pdf_path).name,
                'total_pages': total_pages,
                'content_items': all_content_items,
                'message': f'Successfully extracted {len(tables)} tables from {total_pages} pages'
            }
        
        except Exception as e:
            return {
                'success': False,
                'file_name': Path(pdf_path).name if pdf_path else 'unknown',
                'total_pages': 0,
                'content_items': [],
                'message': f'Extraction failed: {str(e)}'
            }
    
    @staticmethod
    def _is_table_content(line: str, table_content_set: set) -> bool:
        """Detect if a text line is table content"""
        if line in table_content_set:
            return True
        
        # Pure numbers (likely from table rows)
        if re.match(r'^[\d,\.]+$', line):
            return True
        
        # Currency amounts or years
        if re.match(r'^[\$\d,\.\s%]+$', line):
            return True
        
        # Years alone
        if re.match(r'^\d{4}$', line):
            return True
        
        # Known table field labels
        table_labels = [
            'Subscription', 'Professional', 'Product Sales',
            'Cost of Revenue', 'Research & Development',
            'Sales & Marketing', 'General & Administrative',
            'Cash and Cash Equivalents', 'Accounts Receivable',
            'Property and Equipment', 'Intangible Assets',
            'Total Revenue', 'Total Assets'
        ]
        if any(label in line for label in table_labels):
            return True
        
        # Table headers with ($M) or ($)
        if '($M)' in line or '($)' in line:
            return True
        
        return False
