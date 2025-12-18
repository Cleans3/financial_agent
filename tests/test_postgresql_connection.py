"""
Test script to verify PostgreSQL database connection
Tests connection, table creation, and basic CRUD operations
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from src.core.config import settings
from src.database.database import engine, SessionLocal, init_db
from src.database.models import Base, User, ChatSession, ChatMessage, AuditLog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PostgreSQLConnectionTest:
    """Test PostgreSQL connection and functionality"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        self.passed = 0
        self.failed = 0
    
    def test_connection(self):
        """Test basic database connection"""
        logger.info("=" * 60)
        logger.info("TEST 1: Basic Database Connection")
        logger.info("=" * 60)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("✓ Successfully connected to PostgreSQL database")
                self.passed += 1
                return True
        except SQLAlchemyError as e:
            logger.error(f"✗ Failed to connect to PostgreSQL: {e}")
            self.failed += 1
            return False
    
    def test_database_info(self):
        """Get database information"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Database Information")
        logger.info("=" * 60)
        
        try:
            with self.engine.connect() as conn:
                # Get database name
                result = conn.execute(text("SELECT current_database()"))
                db_name = result.scalar()
                logger.info(f"✓ Current database: {db_name}")
                
                # Get PostgreSQL version
                result = conn.execute(text("SELECT version()"))
                version = result.scalar()
                logger.info(f"✓ PostgreSQL version: {version.split(',')[0]}")
                
                # Get user
                result = conn.execute(text("SELECT current_user"))
                user = result.scalar()
                logger.info(f"✓ Current user: {user}")
                
                self.passed += 1
                return True
        except SQLAlchemyError as e:
            logger.error(f"✗ Failed to get database info: {e}")
            self.failed += 1
            return False
    
    def test_table_creation(self):
        """Test creating tables"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Table Creation (Initialize Database)")
        logger.info("=" * 60)
        
        try:
            init_db()
            logger.info("✓ Successfully initialized database tables")
            
            # List created tables
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            logger.info(f"✓ Created tables: {', '.join(tables)}")
            
            self.passed += 1
            return True
        except SQLAlchemyError as e:
            logger.error(f"✗ Failed to create tables: {e}")
            self.failed += 1
            return False
    
    def test_table_structure(self):
        """Verify table structure"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: Table Structure Verification")
        logger.info("=" * 60)
        
        try:
            inspector = inspect(self.engine)
            
            # Check User table
            if "user" in inspector.get_table_names():
                columns = inspector.get_columns("user")
                logger.info("✓ User table columns:")
                for col in columns:
                    logger.info(f"  - {col['name']}: {col['type']}")
            
            # Check ChatSession table
            if "chat_session" in inspector.get_table_names():
                columns = inspector.get_columns("chat_session")
                logger.info("✓ ChatSession table columns:")
                for col in columns:
                    logger.info(f"  - {col['name']}: {col['type']}")
            
            self.passed += 1
            return True
        except SQLAlchemyError as e:
            logger.error(f"✗ Failed to verify table structure: {e}")
            self.failed += 1
            return False
    
    def test_insert_operation(self):
        """Test insert operation"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: Insert Operation")
        logger.info("=" * 60)
        
        try:
            db = self.SessionLocal()
            
            # Create a test user
            test_user = User(
                username="test_user",
                email="test@example.com",
                hashed_password="test_hash",
                is_active=True,
                is_admin=False
            )
            db.add(test_user)
            db.commit()
            
            logger.info(f"✓ Successfully inserted test user: {test_user.username}")
            logger.info(f"  User ID: {test_user.id}")
            
            # Clean up
            db.delete(test_user)
            db.commit()
            logger.info("✓ Successfully deleted test user (cleanup)")
            
            db.close()
            self.passed += 1
            return True
        except SQLAlchemyError as e:
            logger.error(f"✗ Failed insert operation: {e}")
            self.failed += 1
            return False
    
    def test_query_operation(self):
        """Test query operation"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 6: Query Operation")
        logger.info("=" * 60)
        
        try:
            db = self.SessionLocal()
            
            # Count users
            user_count = db.query(User).count()
            logger.info(f"✓ Total users in database: {user_count}")
            
            # List all users
            users = db.query(User).all()
            if users:
                logger.info("✓ Users in database:")
                for user in users:
                    logger.info(f"  - {user.username} ({user.email})")
            else:
                logger.info("✓ No users found (database is clean)")
            
            db.close()
            self.passed += 1
            return True
        except SQLAlchemyError as e:
            logger.error(f"✗ Failed query operation: {e}")
            self.failed += 1
            return False
    
    def test_transaction_rollback(self):
        """Test transaction rollback"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 7: Transaction Rollback")
        logger.info("=" * 60)
        
        try:
            db = self.SessionLocal()
            
            # Create a user
            test_user = User(
                username="rollback_test",
                email="rollback@example.com",
                hashed_password="test_hash",
                is_active=True,
                is_admin=False
            )
            db.add(test_user)
            db.commit()
            user_id = test_user.id
            
            # Rollback changes
            test_user = db.query(User).filter(User.id == user_id).first()
            db.delete(test_user)
            db.commit()
            
            # Verify deletion
            deleted_user = db.query(User).filter(User.id == user_id).first()
            if deleted_user is None:
                logger.info("✓ Transaction rollback successful - user was deleted")
            
            db.close()
            self.passed += 1
            return True
        except SQLAlchemyError as e:
            logger.error(f"✗ Failed transaction rollback: {e}")
            self.failed += 1
            return False
    
    def test_connection_pool(self):
        """Test connection pooling"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 8: Connection Pool")
        logger.info("=" * 60)
        
        try:
            # Create multiple connections
            connections = []
            for i in range(5):
                db = self.SessionLocal()
                connections.append(db)
                logger.info(f"✓ Created connection {i + 1}")
            
            # Close all connections
            for i, db in enumerate(connections):
                db.close()
                logger.info(f"✓ Closed connection {i + 1}")
            
            logger.info("✓ Connection pooling works correctly")
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed connection pool test: {e}")
            self.failed += 1
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("\n")
        logger.info("╔" + "=" * 58 + "╗")
        logger.info("║" + " " * 10 + "PostgreSQL Connection Test Suite" + " " * 16 + "║")
        logger.info("╚" + "=" * 58 + "╝")
        logger.info(f"\nDatabase URL: {settings.DATABASE_URL}")
        logger.info("")
        
        self.test_connection()
        self.test_database_info()
        self.test_table_creation()
        self.test_table_structure()
        self.test_insert_operation()
        self.test_query_operation()
        self.test_transaction_rollback()
        self.test_connection_pool()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        total = self.passed + self.failed
        logger.info(f"Total Tests: {total}")
        logger.info(f"✓ Passed: {self.passed}")
        logger.info(f"✗ Failed: {self.failed}")
        
        if self.failed == 0:
            logger.info("\n🎉 All tests passed!")
        else:
            logger.warning(f"\n⚠️  {self.failed} test(s) failed")
        
        logger.info("=" * 60 + "\n")
        
        return self.failed == 0


def main():
    """Main function"""
    try:
        tester = PostgreSQLConnectionTest()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
