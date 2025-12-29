"""
Table extraction and repair utilities.
Handles multi-row cells, missing headers, and structural validation.
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional


class TableRepairer:
    """Repair and validate extracted tables"""
    
    @staticmethod
    def is_likely_header_row(row: List) -> bool:
        """Check if row contains header-like values (short, no numbers, meaningful)"""
        if not row or not any(row):
            return False
        
        # Header keywords
        header_keywords = ['2024', '2023', '2022', '2021', 'change', 'basis', 'period', 
                          'end of', 'year', 'balances', 'december', 'ratio', 'capital',
                          'approach', 'standardized', 'advanced']
        
        text_content = ' '.join(str(cell).lower() for cell in row if cell)
        
        # Check if contains typical header keywords
        has_year = any(year in text_content for year in ['2024', '2023', '2022', '2021'])
        has_header_keyword = any(kw in text_content for kw in header_keywords)
        
        return has_year or has_header_keyword
    
    @staticmethod
    def is_likely_title_row(row: List) -> bool:
        """Check if row is a page title/header, not actual data"""
        if not row or len(row) == 0:
            return False
        
        # Title rows typically have very few columns with content and are spaced strangely
        first_cell = str(row[0]).upper() if row[0] else ""
        
        # Examples: "ANNUAL REPORT", "FINANCIAL HIGHLIGHTS", etc.
        title_keywords = ['annual', 'financial', 'highlights', 'report', 'quarter']
        
        # If first cell has title keywords and is oddly spaced, it's a title
        if any(kw in first_cell for kw in title_keywords):
            # Check if letters are spaced out (sign of OCR of large title)
            spaces = first_cell.count(' ')
            if spaces > len(first_cell) / 2:  # More than 50% spaces
                return True
        
        return False
    
    @staticmethod
    def merge_currency_with_value(table: List[List]) -> List[List]:
        """Merge rows where currency symbols are separated from values
        
        Pattern 1 - Currency in middle columns with value in next row's first cols:
        Row N:   ['Label...', '$', '$', '22%']
        Row N+1: ['8,322.2', '6,831.0', None, None]
        
        Should become:
        Row N:   ['Label...', '$8,322.2', '$6,831.0', '22%']
        (skip Row N+1)
        """
        if len(table) < 2:
            return table
        
        merged = []
        skip_next = False
        
        for i, current_row in enumerate(table):
            # Check if we should skip this row (was merged into previous)
            if skip_next:
                skip_next = False
                continue
            
            # Try to detect and merge with next row
            if i + 1 < len(table):
                next_row = table[i + 1]
                
                # Pattern: current row has label + mostly '$' | next row has numbers in first cols
                dollar_cols = sum(1 for j, cell in enumerate(current_row[1:], 1) 
                                if isinstance(cell, str) and cell.strip() == '$')
                
                # Check if at least 2 $ symbols (excluding first col)
                if dollar_cols >= 2 and len(current_row) >= 3:
                    # Check next row has numbers
                    number_count = 0
                    for cell in next_row[:-1]:  # Check all but last
                        if cell:
                            try:
                                float(str(cell).replace(',', '').replace('$', '').strip())
                                number_count += 1
                            except (ValueError, AttributeError):
                                pass
                    
                    # If next row has mostly numbers, merge
                    if number_count >= len(current_row) - 2:  # Most columns have numbers
                        merged_row = [current_row[0]]  # Label
                        
                        # Merge currency with values
                        for j in range(1, len(current_row)):
                            curr_cell = current_row[j]
                            next_cell = next_row[j-1] if j-1 < len(next_row) else None
                            
                            if curr_cell == '$':
                                # $ symbol - prepend to next value
                                if next_cell:
                                    merged_row.append(f"${next_cell}")
                                else:
                                    merged_row.append('$')
                            elif '%' in str(curr_cell or ''):
                                # Keep percentage
                                merged_row.append(curr_cell)
                            elif next_cell:
                                # Use value from next row
                                merged_row.append(next_cell)
                            else:
                                merged_row.append(curr_cell)
                        
                        merged.append(merged_row)
                        skip_next = True
                        continue
            
            # No merge - just add the row
            merged.append(current_row)
        
        return merged
    
    @staticmethod
    def split_multiline_cells(table: List[List]) -> List[List]:
        """
        Split cells containing newlines into separate rows.
        Handles: ['$ 8,322.2\n2,031.1\n608.4'] → separate rows
        """
        if not table:
            return table
        
        # Find maximum lines in any cell
        max_lines = 1
        for row in table:
            for cell in row:
                if cell:
                    lines = str(cell).split('\n')
                    max_lines = max(max_lines, len(lines))
        
        if max_lines == 1:
            return table  # No multiline cells
        
        # Expand rows
        expanded_table = []
        for row in table:
            expanded_rows = [[] for _ in range(max_lines)]
            
            for cell in row:
                if cell:
                    lines = str(cell).split('\n')
                    for line_idx, line in enumerate(lines):
                        expanded_rows[line_idx].append(line.strip() if line else '')
                else:
                    for line_idx in range(max_lines):
                        expanded_rows[line_idx].append(None)
            
            expanded_table.extend(expanded_rows)
        
        return expanded_table
    
    @staticmethod
    def remove_empty_rows_cols(table: List[List]) -> List[List]:
        """Remove completely empty rows and columns"""
        if not table:
            return table
        
        # Remove empty rows
        table = [row for row in table if any(cell for cell in row)]
        
        if not table:
            return table
        
        # Remove empty columns
        col_count = len(table[0])
        cols_to_keep = []
        
        for col_idx in range(col_count):
            if any(table[row_idx][col_idx] for row_idx in range(len(table))):
                cols_to_keep.append(col_idx)
        
        table = [[row[idx] if idx < len(row) else None for idx in cols_to_keep] 
                 for row in table]
        
        return table
    
    @staticmethod
    def validate_column_count(table: List[List]) -> List[List]:
        """Ensure all rows have same column count"""
        if not table:
            return table
        
        max_cols = max(len(row) for row in table) if table else 0
        
        for row in table:
            while len(row) < max_cols:
                row.append(None)
        
        return table
    
    @staticmethod
    def remove_preamble_rows(table: List[List]) -> List[List]:
        """Remove text rows that appear above the actual table (Camelot artifact)
        
        Camelot sometimes captures narrative text above tables as Row 0.
        These rows have characteristics:
        - Contain long text passages (>50 chars)
        - Very different structure from Row 1
        - Are clearly not part of the data structure
        
        Examples of preamble to remove:
        - "from 30 to 90 days, and no significant financing component is deemed present."
        - "marketing programs, legal fees, and professional services."
        - "4. Consolidated Balance Sheets" (when it's just context, not a header)
        """
        if len(table) < 2:
            return table
        
        row0 = table[0]
        row1 = table[1] if len(table) > 1 else None
        
        # Check if Row 0 is likely a preamble
        # Characteristic 1: First cell is long text (likely a sentence fragment)
        first_cell = str(row0[0]).strip() if row0 and row0[0] else ""
        
        # Characteristic 2: Structure mismatch - Row 1 looks more like table data
        is_preamble = False
        
        if len(first_cell) > 50:  # Long text passage
            # Check if row1 looks more like actual data
            if row1 and any(row1):
                # Row 1 has headers or structured data
                row1_text = ' '.join(str(c).lower() for c in row1 if c)
                
                # If Row 1 contains year columns or header keywords
                if any(year in row1_text for year in ['2024', '2023', '2022', '2021']):
                    is_preamble = True
                elif any(kw in row1_text for kw in ['revenue', 'assets', 'expense', 'income', 'ratio']):
                    is_preamble = True
        
        # Characteristic 3: Row 0 is a sentence fragment (contains common endings)
        if not is_preamble and (first_cell.endswith('.') or 
                                first_cell.endswith(',') or 
                                'and' in first_cell or
                                'the' in first_cell.lower()):
            # It's likely narrative text, check if Row 1 is more structured
            if row1:
                row1_str = ' '.join(str(c) for c in row1 if c)
                # If row1 is short and contains year/numbers, row0 is preamble
                if len(row1_str) < 60 and any(c.isdigit() for c in row1_str):
                    is_preamble = True
        
        if is_preamble:
            return table[1:]
        
        return table
    
    @staticmethod
    def fix_missing_headers(table: List[List]) -> List[List]:
        """Detect and fix misaligned headers and orphaned data rows
        
        Fixes:
        1. Remove section header rows (contain only text + empty cells, e.g., "4. Consolidated Balance Sheets")
        2. Header row alignment: [values...] → [None, values...] (but only if consistent with data rows)
        3. Orphaned data rows that belong to previous row
        """
        if len(table) < 1:
            return table
        
        # Fix 0: Remove section header rows (like "4. Consolidated Balance Sheets" with empty cells)
        # Pattern: Row contains one text cell followed by empty cells
        while len(table) > 1:
            row0 = table[0]
            # Check if row0 is a section header: first cell has text, rest are empty
            has_section_header = (
                row0 and 
                len(row0) >= 2 and
                row0[0] and 
                str(row0[0]).strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) and
                all(not c or (isinstance(c, str) and not c.strip()) for c in row0[1:])
            )
            if has_section_header:
                table = table[1:]
            else:
                break
        
        if not table:
            return table
        
        header = table[0]
        
        # Fix 1: Ensure header and data rows have same column count
        # If header has extra None at start that data rows don't have, remove it
        if len(table) > 1:
            data_row_cols = len(table[1])
            header_cols = len(header)
            
            # If header has more columns than data rows
            if header_cols > data_row_cols:
                # Check if it starts with None (spurious added column)
                if header[0] is None:
                    # Remove the extra None to match data rows
                    table[0] = header[1:]
        
        # Fix 2: Remove completely empty header row
        if header and not any(h for h in header):
            table = table[1:]
            if not table:
                return table
        
        # Fix 3: Merge orphaned data rows into previous row
        # Pattern: Some rows have values only in first column (no label)
        # These should be part of the previous row
        fixed = []
        i = 0
        
        while i < len(table):
            current_row = list(table[i])
            
            # Check if next row looks like orphaned data
            if i + 1 < len(table):
                next_row = table[i + 1]
                
                # Orphaned row: starts with single value, rest mostly empty
                is_orphaned = (
                    next_row and len(next_row) > 0 and
                    next_row[0] and  # Has value in column 0
                    (not isinstance(next_row[0], str) or 
                     not any(char.isalpha() for char in str(next_row[0])))  # Numeric value
                    and sum(1 for cell in next_row[1:] if cell) <= 1  # Rest mostly empty
                )
                
                # If current row ends with numeric values and next is orphaned
                if is_orphaned and current_row[-1]:
                    try:
                        float(str(current_row[-1]).replace(',', '').replace('%', '').strip())
                        # Current row is numeric, next row is orphaned single value
                        # Merge them by extending current row
                        merged_row = list(current_row)
                        merged_row.extend(next_row)
                        fixed.append(merged_row)
                        i += 2  # Skip both rows
                        continue
                    except (ValueError, AttributeError):
                        pass
            
            fixed.append(current_row)
            i += 1
        
        return fixed
    
    @staticmethod
    def filter_non_table_content(table: List[List]) -> Optional[List[List]]:
        """Filter out page titles and non-table content
        
        Returns None if table is determined to be a title/header, not actual data
        """
        if not table or len(table) == 0:
            return table
        
        # Check if this looks like a page title table (1 column with spaced letters)
        if len(table[0]) == 1:
            # Single column - check ALL rows to see if they look like scattered titles
            title_score = 0
            
            for row in table:
                if not row or not row[0]:
                    continue
                    
                content = str(row[0]).strip().upper()
                spaces = content.count(' ')
                total_chars = len(content)
                
                if total_chars == 0:
                    continue
                
                space_ratio = spaces / total_chars
                
                # High space ratio indicates spaced out text (OCR of large title)
                # Look for keywords that could be scattered across spaces
                has_keywords = any(kw in content.upper() for kw in 
                                  ['ANNUAL', 'REPORT', 'FINANCIAL', 'HIGHLIGHTS', 
                                   'QUARTER', 'SUMMARY', 'OVERVIEW'])
                
                # Also check if it's spaced-out version of keywords
                # e.g., "A N N UA L" contains "ANNUAL" as individual letters
                spaced_keywords = any(
                    all(char in content for char in kw)  # All letters of keyword present
                    for kw in ['ANNUAL', 'REPORT', 'FINANCIAL', 'HIGHLIGHTS']
                )
                
                is_title_like = space_ratio > 0.30  # High space ratio
                
                if is_title_like and (has_keywords or spaced_keywords):
                    title_score += 1
            
            # If majority of rows look like title rows, this is a title table
            if title_score >= len(table) * 0.5:  # 50% or more rows are title-like
                return None
        
        return table
    
    @staticmethod
    def repair_table(table: List[List]) -> Optional[List[List]]:
        """Apply all repairs in sequence. Returns None if table is not actual data."""
        if not table:
            return table
        
        # First, filter out non-table content (titles, etc)
        table = TableRepairer.filter_non_table_content(table)
        if table is None:
            return None
        
        # Remove preamble rows (narrative text captured above table)
        table = TableRepairer.remove_preamble_rows(table)
        
        # Order matters!
        table = TableRepairer.split_multiline_cells(table)
        table = TableRepairer.validate_column_count(table)
        table = TableRepairer.merge_currency_with_value(table)
        table = TableRepairer.remove_empty_rows_cols(table)
        table = TableRepairer.fix_missing_headers(table)
        
        return table


class TableAnalyzer:
    """Analyze table structure and characteristics"""
    
    @staticmethod
    def get_table_info(table: List[List]) -> Dict:
        """Extract metadata about table structure"""
        if not table:
            return {
                'rows': 0,
                'cols': 0,
                'has_headers': False,
                'numeric_cols': [],
                'text_cols': []
            }
        
        rows = len(table)
        cols = len(table[0]) if table else 0
        
        # Analyze column types
        numeric_cols = []
        text_cols = []
        
        for col_idx in range(cols):
            col_data = [table[row_idx][col_idx] for row_idx in range(rows)]
            is_numeric = TableAnalyzer._is_numeric_column(col_data)
            
            if is_numeric:
                numeric_cols.append(col_idx)
            else:
                text_cols.append(col_idx)
        
        return {
            'rows': rows,
            'cols': cols,
            'numeric_cols': numeric_cols,
            'text_cols': text_cols,
            'has_headers': rows > 1,
            'first_col_is_label': 0 in text_cols
        }
    
    @staticmethod
    def _is_numeric_column(col_data: List) -> bool:
        """Check if column contains mostly numeric values"""
        numeric_count = 0
        valid_count = 0
        
        for cell in col_data:
            if cell is None or cell == '':
                continue
            
            valid_count += 1
            cell_str = str(cell).strip()
            
            # Try to parse as number
            try:
                # Remove common financial characters
                cleaned = cell_str.replace('$', '').replace(',', '').replace('%', '')
                float(cleaned)
                numeric_count += 1
            except ValueError:
                pass
        
        if valid_count == 0:
            return False
        
        return (numeric_count / valid_count) > 0.7
    
    @staticmethod
    def extract_key_metrics(table: List[List]) -> List[str]:
        """Extract row labels (first column) as key metrics"""
        if not table or len(table[0]) == 0:
            return []
        
        metrics = []
        for row in table:
            if row[0]:  # First column
                metrics.append(str(row[0]).strip())
        
        return metrics


class TableExtractor:
    """Extract tables using camelot"""
    
    @staticmethod
    def extract_tables(pdf_path: str, use_camelot: bool = True) -> List[Dict]:
        """
        Extract tables from PDF using camelot or pdfplumber fallback
        
        Args:
            pdf_path: Path to PDF
            use_camelot: Try camelot first, fallback to pdfplumber
        
        Returns:
            List of dicts with 'data' (list of lists) and 'page' (page number)
        """
        tables = []
        
        if use_camelot:
            tables = TableExtractor._extract_with_camelot(pdf_path)
        
        if not tables:
            tables = TableExtractor._extract_with_pymupdf(pdf_path)
        
        return tables
    
    @staticmethod
    def _extract_with_camelot(pdf_path: str) -> List[Dict]:
        """Extract using camelot (better for structured tables)"""
        try:
            import camelot
        except ImportError:
            print("camelot not installed, skipping camelot extraction")
            return []
        
        tables = []
        
        try:
            # Try stream flavor FIRST (better for financial reports with text-formatted tables)
            camelot_tables = camelot.read_pdf(
                pdf_path,
                pages='all',
                flavor='stream',
                suppress_stdout=True
            )
            
            # If stream didn't find tables, try lattice
            if not camelot_tables:
                camelot_tables = camelot.read_pdf(
                    pdf_path,
                    pages='all',
                    flavor='lattice',
                    suppress_stdout=True
                )
            
            for table in camelot_tables:
                table_data = table.df.values.tolist()
                table_data = TableRepairer.repair_table(table_data)
                
                # Skip if table was filtered out (e.g., page titles)
                if table_data is None:
                    continue
                
                tables.append({
                    'data': table_data,
                    'page': table.page,
                    'source': 'camelot'
                })
        except Exception as e:
            print(f"Camelot extraction failed: {e}")
            return []
        
        return tables
    
    @staticmethod
    def _extract_with_pymupdf(pdf_path: str) -> List[Dict]:
        """Fallback to pymupdf table extraction"""
        import pymupdf
        
        tables = []
        doc = pymupdf.open(pdf_path)
        
        try:
            for page_num, page in enumerate(doc):
                # pymupdf doesn't have native table extraction
                # Just return empty for now - tables will be handled as text
                pass
        finally:
            doc.close()
        
        return tables
