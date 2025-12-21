"""Add document_uploads tracking table

Revision ID: 8f8c9a1b2c3d
Revises: 707bbe04f7f7
Create Date: 2025-12-21 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '8f8c9a1b2c3d'
down_revision: Union[str, Sequence[str], None] = '707bbe04f7f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'document_uploads',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('doc_id', sa.String(), nullable=False),
        sa.Column('uploaded_by_admin_id', sa.String(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('chunk_count', sa.Integer(), nullable=True),
        sa.Column('extraction_time_ms', sa.Integer(), nullable=True),
        sa.Column('embedding_time_ms', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('tags', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_document_uploads_doc_id'), 'document_uploads', ['doc_id'], unique=False)
    op.create_index(op.f('ix_document_uploads_admin_id'), 'document_uploads', ['uploaded_by_admin_id'], unique=False)
    op.create_index(op.f('ix_document_uploads_status'), 'document_uploads', ['status'], unique=False)
    op.create_index(op.f('ix_document_uploads_created_at'), 'document_uploads', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_document_uploads_created_at'), table_name='document_uploads')
    op.drop_index(op.f('ix_document_uploads_status'), table_name='document_uploads')
    op.drop_index(op.f('ix_document_uploads_admin_id'), table_name='document_uploads')
    op.drop_index(op.f('ix_document_uploads_doc_id'), table_name='document_uploads')
    op.drop_table('document_uploads')
