"""
Table chunking service - creates semantic chunks from financial tables.
Different logic than text chunking - preserves table structure.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TableChunk:
    """A chunk representing part or all of a table"""
    chunk_id: str
    content: str  # Markdown representation
    table_data: List[List]  # Raw table data
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dict for storage"""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'metadata': self.metadata
        }


class TableChunkingService:
    """Create logical chunks from tables with rich metadata"""
    
    def __init__(self, max_rows_per_chunk: int = 15):
        """
        Args:
            max_rows_per_chunk: Maximum rows per chunk (tables split if larger)
        """
        self.max_rows_per_chunk = max_rows_per_chunk
        self.chunk_counter = 0
    
    def chunk_table(
        self,
        table_data: List[List],
        table_name: str,
        document_metadata: Dict,
        page_number: int
    ) -> List[TableChunk]:
        """
        Create chunks from a table while preserving structure.
        
        Args:
            table_data: List of lists (rows × cols)
            table_name: Name of the table
            document_metadata: Document context (company, year, etc.)
            page_number: Page number in document
        
        Returns:
            List of TableChunk objects
        """
        if not table_data:
            return []
        
        chunks = []
        rows = len(table_data)
        
        # If table is small, keep as single chunk
        if rows <= self.max_rows_per_chunk:
            chunk = self._create_single_chunk(
                table_data=table_data,
                table_name=table_name,
                document_metadata=document_metadata,
                page_number=page_number,
                row_range=(0, rows),
                total_rows=rows
            )
            chunks.append(chunk)
        else:
            # Split large tables
            for start_row in range(0, rows, self.max_rows_per_chunk):
                end_row = min(start_row + self.max_rows_per_chunk, rows)
                chunk_table_data = table_data[start_row:end_row]
                
                chunk = self._create_single_chunk(
                    table_data=chunk_table_data,
                    table_name=table_name,
                    document_metadata=document_metadata,
                    page_number=page_number,
                    row_range=(start_row, end_row),
                    total_rows=rows
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_single_chunk(
        self,
        table_data: List[List],
        table_name: str,
        document_metadata: Dict,
        page_number: int,
        row_range: tuple,
        total_rows: int
    ) -> TableChunk:
        """Create a single table chunk with complete metadata"""
        
        self.chunk_counter += 1
        chunk_id = f"table_chunk_{self.chunk_counter}"
        
        # Convert table to markdown
        markdown_content = self._table_to_markdown(table_data)
        
        # Build rich metadata
        metadata = self._build_metadata(
            chunk_id=chunk_id,
            table_name=table_name,
            table_data=table_data,
            document_metadata=document_metadata,
            page_number=page_number,
            row_range=row_range,
            total_rows=total_rows
        )
        
        return TableChunk(
            chunk_id=chunk_id,
            content=markdown_content,
            table_data=table_data,
            metadata=metadata
        )
    
    def _table_to_markdown(self, table_data: List[List]) -> str:
        """Convert table to markdown format"""
        if not table_data:
            return ""
        
        lines = []
        
        for row in table_data:
            # Convert cells to strings, handle None
            cells = [str(cell) if cell is not None else '' for cell in row]
            line = '| ' + ' | '.join(cells) + ' |'
            lines.append(line)
            
            # Add separator after first row (header)
            if len(lines) == 1:
                separator = '|' + '|'.join(['---' for _ in row]) + '|'
                lines.append(separator)
        
        return '\n'.join(lines)
    
    def _build_metadata(
        self,
        chunk_id: str,
        table_name: str,
        table_data: List[List],
        document_metadata: Dict,
        page_number: int,
        row_range: tuple,
        total_rows: int
    ) -> Dict:
        """Build comprehensive metadata for chunk"""
        
        # Extract key information from table
        columns = table_data[0] if table_data else []
        key_metrics = self._extract_key_metrics(table_data)
        
        return {
            # Identification
            "chunk_id": chunk_id,
            "chunk_type": "table",
            "table_name": table_name,
            
            # Source context
            "document": document_metadata.get("document", ""),
            "company": document_metadata.get("company", ""),
            "fiscal_year": document_metadata.get("fiscal_year"),
            "page_number": page_number,
            
            # Table specifics
            "row_range": list(row_range),
            "total_rows": total_rows,
            "total_cols": len(columns) if columns else 0,
            "columns": [str(c) for c in columns] if columns else [],
            "is_complete_table": (row_range[1] >= total_rows),
            
            # Content classification
            "section": document_metadata.get("section", ""),
            "subsection": document_metadata.get("subsection", ""),
            "data_type": document_metadata.get("data_type", "financial"),
            
            # Search hints
            "key_metrics": key_metrics,
            "contains_growth_rates": self._detect_growth_rates(table_data),
            "contains_ratios": self._detect_ratios(table_data),
            "contains_percentages": self._detect_percentages(table_data),
            
            # Relationships
            "related_tables": document_metadata.get("related_tables", []),
            "parent_section": document_metadata.get("parent_section", ""),
            
            # Description
            "description": self._generate_description(
                table_name, row_range, total_rows
            )
        }
    
    def _extract_key_metrics(self, table_data: List[List]) -> List[str]:
        """Extract row labels from first column"""
        if not table_data or not table_data[0]:
            return []
        
        metrics = []
        for row in table_data:
            if row and row[0]:
                metric = str(row[0]).strip()
                if metric and len(metric) > 0:
                    metrics.append(metric)
        
        return metrics[:20]  # Limit to first 20
    
    def _detect_growth_rates(self, table_data: List[List]) -> bool:
        """Check if table contains growth rate percentages"""
        text = ' '.join(str(cell) for row in table_data for cell in row if cell)
        return any(phrase in text.lower() for phrase in [
            'growth', 'increase', 'decrease', 'change',
            'yoy', 'qoq', 'vs', '%'
        ])
    
    def _detect_ratios(self, table_data: List[List]) -> bool:
        """Check if table contains financial ratios"""
        text = ' '.join(str(cell) for row in table_data for cell in row if cell)
        return any(phrase in text.lower() for phrase in [
            'ratio', 'roe', 'roa', 'roic', 'debt', 'equity',
            'margin', 'current', 'quick', 'leverage'
        ])
    
    def _detect_percentages(self, table_data: List[List]) -> bool:
        """Check if table contains percentage values"""
        text = ' '.join(str(cell) for row in table_data for cell in row if cell)
        return '%' in text
    
    def _generate_description(
        self,
        table_name: str,
        row_range: tuple,
        total_rows: int
    ) -> str:
        """Generate human-readable description"""
        if row_range[1] >= total_rows:
            return f"{table_name} (complete table)"
        else:
            return f"{table_name} - Rows {row_range[0]}-{row_range[1]} of {total_rows}"
