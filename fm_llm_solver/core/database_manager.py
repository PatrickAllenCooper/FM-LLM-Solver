"""
Production database manager for FM-LLM-Solver.

Provides PostgreSQL connection management, migrations, connection pooling,
and schema management for production deployments.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator

try:
    import asyncpg
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor
    from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, Boolean, Text, JSON
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
    from sqlalchemy.pool import QueuePool
    from alembic import command
    from alembic.config import Config
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

from .config_manager import ConfigurationManager
from .logging_manager import get_logger
from .exceptions import DatabaseError, ConfigurationError


# Define Base class only if SQLAlchemy is available
if HAS_POSTGRES:
    class Base(DeclarativeBase):
        """Base class for SQLAlchemy models."""
        pass
else:
    # Fallback base class when SQLAlchemy is not available
    class Base:
        """Fallback base class when SQLAlchemy is not available."""
        pass


class DatabaseManager:
    """
    Production database manager with PostgreSQL support.
    
    Features:
    - Connection pooling with configurable pool size
    - Async and sync connection support
    - Database migrations with Alembic
    - Schema management and validation
    - Health checks and monitoring
    - Backup and restore capabilities
    - Multi-database support
    """
    
    def __init__(
        self,
        config_manager: Optional[ConfigurationManager] = None,
        db_name: str = "primary"
    ):
        """
        Initialize database manager.
        
        Args:
            config_manager: Configuration manager instance
            db_name: Database configuration name
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.db_name = db_name
        self.logger = get_logger(f"database.{db_name}")
        
        # Configuration
        self.db_config = self._get_database_config()
        
        if HAS_POSTGRES:
            self.connection_string = self._build_connection_string()
            self.async_connection_string = self._build_async_connection_string()
            
            # Connection pools
            self._sync_pool: Optional[pool.ThreadedConnectionPool] = None
            self._async_engine = None
            self._sync_engine = None
            
            # Session makers
            self._async_session_maker = None
            self._sync_session_maker = None
        
        # Migration configuration
        self.migrations_dir = Path("migrations")
        self.migrations_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._initialized = False
        
        self.logger.info(f"Database manager initialized for {db_name}")
    
    def _get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        try:
            config = self.config_manager.load_config()
            databases = config.get('database', {})
            
            if self.db_name not in databases:
                # Return default configuration
                return {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'fm_llm_solver',
                    'username': 'postgres',
                    'password': os.environ.get('DB_PASSWORD', ''),
                    'pool_size': 20,
                    'max_overflow': 10,
                    'pool_timeout': 30,
                    'pool_recycle': 3600,
                    'ssl_mode': 'prefer'
                }
            
            return databases[self.db_name]
        except Exception as e:
            self.logger.error(f"Failed to load database config: {e}")
            # Return default configuration
            return {
                'host': 'localhost',
                'port': 5432,
                'database': 'fm_llm_solver',
                'username': 'postgres',
                'password': os.environ.get('DB_PASSWORD', ''),
                'pool_size': 20,
                'max_overflow': 10,
                'pool_timeout': 30,
                'pool_recycle': 3600,
                'ssl_mode': 'prefer'
            }
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        config = self.db_config
        
        # Extract connection parameters
        host = config.get('host', 'localhost')
        port = config.get('port', 5432)
        database = config.get('database', 'fm_llm_solver')
        username = config.get('username', 'postgres')
        password = config.get('password', '')
        ssl_mode = config.get('ssl_mode', 'prefer')
        
        # Build connection string
        conn_str = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        
        # Add SSL parameters
        if ssl_mode:
            conn_str += f"?sslmode={ssl_mode}"
        
        return conn_str
    
    def _build_async_connection_string(self) -> str:
        """Build async PostgreSQL connection string."""
        return self.connection_string.replace('postgresql://', 'postgresql+asyncpg://')
    
    async def initialize(self) -> None:
        """Initialize database connections and run migrations."""
        if self._initialized:
            return
        
        if not HAS_POSTGRES:
            self.logger.warning("PostgreSQL dependencies not installed, using fallback mode")
            self._initialized = True
            return
        
        try:
            # Create engines
            await self._create_engines()
            
            # Create connection pools
            await self._create_connection_pools()
            
            # Run health check
            await self._health_check()
            
            # Run migrations
            await self._run_migrations()
            
            self._initialized = True
            self.logger.info("Database manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    async def _create_engines(self) -> None:
        """Create SQLAlchemy engines."""
        pool_config = {
            'pool_size': self.db_config.get('pool_size', 20),
            'max_overflow': self.db_config.get('max_overflow', 10),
            'pool_timeout': self.db_config.get('pool_timeout', 30),
            'pool_recycle': self.db_config.get('pool_recycle', 3600),
            'pool_pre_ping': True,
            'poolclass': QueuePool,
        }
        
        # Async engine
        self._async_engine = create_async_engine(
            self.async_connection_string,
            echo=self.db_config.get('echo', False),
            **pool_config
        )
        
        # Sync engine
        self._sync_engine = create_engine(
            self.connection_string,
            echo=self.db_config.get('echo', False),
            **pool_config
        )
        
        # Session makers
        self._async_session_maker = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        self._sync_session_maker = sessionmaker(
            self._sync_engine,
            class_=Session,
            expire_on_commit=False
        )
        
        self.logger.info("Database engines created")
    
    async def _create_connection_pools(self) -> None:
        """Create connection pools for direct database access."""
        config = self.db_config
        
        # Sync connection pool
        self._sync_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=config.get('pool_size', 20),
            host=config.get('host', 'localhost'),
            port=config.get('port', 5432),
            database=config.get('database', 'fm_llm_solver'),
            user=config.get('username', 'postgres'),
            password=config.get('password', ''),
            sslmode=config.get('ssl_mode', 'prefer')
        )
        
        self.logger.info("Connection pools created")
    
    async def _health_check(self) -> None:
        """Perform database health check."""
        try:
            # Test async connection
            async with self._async_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                await result.fetchone()
            
            # Test sync connection
            with self._sync_engine.begin() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            self.logger.info("Database health check passed")
            
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            raise DatabaseError(f"Database health check failed: {e}")
    
    async def _run_migrations(self) -> None:
        """Run database migrations."""
        try:
            # Create alembic configuration
            alembic_cfg = Config()
            alembic_cfg.set_main_option('script_location', str(self.migrations_dir))
            alembic_cfg.set_main_option('sqlalchemy.url', self.connection_string)
            
            # Run migrations
            if self.migrations_dir.exists():
                command.upgrade(alembic_cfg, 'head')
                self.logger.info("Database migrations completed")
            else:
                self.logger.warning("No migrations directory found")
                
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise DatabaseError(f"Migration failed: {e}")
    
    @asynccontextmanager
    async def async_session(self) -> AsyncGenerator["AsyncSession", None]:
        """Get async database session."""
        if not self._initialized:
            await self.initialize()
        
        if not HAS_POSTGRES:
            raise DatabaseError("PostgreSQL not available")
        
        session = self._async_session_maker()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            await session.close()
    
    @contextmanager
    def sync_session(self) -> "Session":
        """Get sync database session."""
        if not self._initialized:
            # Initialize synchronously
            asyncio.run(self.initialize())
        
        if not HAS_POSTGRES:
            raise DatabaseError("PostgreSQL not available")
        
        session = self._sync_session_maker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            session.close()
    
    @contextmanager
    def raw_connection(self):
        """Get raw database connection from pool."""
        if not self._sync_pool:
            raise DatabaseError("Connection pool not initialized")
        
        conn = self._sync_pool.getconn()
        try:
            yield conn
        finally:
            self._sync_pool.putconn(conn)
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        fetch: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute SQL query asynchronously."""
        if not HAS_POSTGRES:
            raise DatabaseError("PostgreSQL not available")
        
        async with self.async_session() as session:
            result = await session.execute(text(query), params or {})
            
            if fetch:
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
            else:
                return []
    
    def execute_query_sync(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        fetch: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute SQL query synchronously."""
        with self.sync_session() as session:
            result = session.execute(text(query), params or {})
            
            if fetch:
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
            else:
                return []
    
    async def create_tables(self, metadata: "MetaData") -> None:
        """Create database tables from metadata."""
        async with self._async_engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        
        self.logger.info("Database tables created")
    
    def create_tables_sync(self, metadata: "MetaData") -> None:
        """Create database tables from metadata synchronously."""
        with self._sync_engine.begin() as conn:
            metadata.create_all(conn)
        
        self.logger.info("Database tables created")
    
    async def backup_database(self, backup_path: Union[str, Path]) -> None:
        """Create database backup."""
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = self.db_config
        
        # Use pg_dump for backup
        cmd = [
            'pg_dump',
            '--host', config.get('host', 'localhost'),
            '--port', str(config.get('port', 5432)),
            '--username', config.get('username', 'postgres'),
            '--dbname', config.get('database', 'fm_llm_solver'),
            '--file', str(backup_path),
            '--format', 'custom',
            '--compress', '9',
            '--verbose'
        ]
        
        # Set password environment variable
        env = os.environ.copy()
        env['PGPASSWORD'] = config.get('password', '')
        
        import subprocess
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DatabaseError(f"Database backup failed: {result.stderr}")
        
        self.logger.info(f"Database backup created: {backup_path}")
    
    async def restore_database(self, backup_path: Union[str, Path]) -> None:
        """Restore database from backup."""
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise DatabaseError(f"Backup file not found: {backup_path}")
        
        config = self.db_config
        
        # Use pg_restore for restore
        cmd = [
            'pg_restore',
            '--host', config.get('host', 'localhost'),
            '--port', str(config.get('port', 5432)),
            '--username', config.get('username', 'postgres'),
            '--dbname', config.get('database', 'fm_llm_solver'),
            '--clean',
            '--if-exists',
            '--verbose',
            str(backup_path)
        ]
        
        # Set password environment variable
        env = os.environ.copy()
        env['PGPASSWORD'] = config.get('password', '')
        
        import subprocess
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DatabaseError(f"Database restore failed: {result.stderr}")
        
        self.logger.info(f"Database restored from: {backup_path}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not HAS_POSTGRES:
            return {'status': 'PostgreSQL not available'}
        
        try:
            stats_query = """
            SELECT 
                current_database() as database_name,
                pg_database_size(current_database()) as size_bytes,
                (SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()) as connections
            """
            
            result = await self.execute_query(stats_query)
            return result[0] if result else {}
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get connection pool information."""
        pool_info = {}
        
        if self._sync_pool:
            pool_info['sync_pool'] = {
                'size': self._sync_pool.minconn,
                'max_size': self._sync_pool.maxconn,
                'available': len(self._sync_pool._pool),
                'used': self._sync_pool.maxconn - len(self._sync_pool._pool)
            }
        
        if self._async_engine:
            pool_info['async_pool'] = {
                'size': self._async_engine.pool.size(),
                'checked_in': self._async_engine.pool.checkedin(),
                'checked_out': self._async_engine.pool.checkedout(),
                'overflow': self._async_engine.pool.overflow(),
                'invalid': self._async_engine.pool.invalid()
            }
        
        return pool_info
    
    async def close(self) -> None:
        """Close database connections."""
        if HAS_POSTGRES:
            if self._async_engine:
                await self._async_engine.dispose()
            
            if self._sync_engine:
                self._sync_engine.dispose()
            
            if self._sync_pool:
                self._sync_pool.closeall()
        
        self.logger.info("Database connections closed")


# Global database manager instances
_database_managers: Dict[str, DatabaseManager] = {}


def get_database_manager(db_name: str = "primary") -> DatabaseManager:
    """Get database manager instance."""
    if db_name not in _database_managers:
        _database_managers[db_name] = DatabaseManager(db_name=db_name)
    
    return _database_managers[db_name]


async def initialize_databases() -> None:
    """Initialize all database managers."""
    for db_manager in _database_managers.values():
        await db_manager.initialize()


async def close_databases() -> None:
    """Close all database connections."""
    for db_manager in _database_managers.values():
        await db_manager.close() 