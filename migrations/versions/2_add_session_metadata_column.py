"""Add session_metadata column to chat_sessions table

Revision ID: 9g9d0e2c3f4e
Revises: 8f8c9a1b2c3d
Create Date: 2025-12-24 19:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '9g9d0e2c3f4e'
down_revision: Union[str, Sequence[str], None] = '8f8c9a1b2c3d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('chat_sessions', 
        sa.Column('session_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True, server_default='{}'))


def downgrade() -> None:
    op.drop_column('chat_sessions', 'session_metadata')
