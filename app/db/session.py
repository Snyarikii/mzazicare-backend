from collections.abc import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.logger import is_development

# SQL echo prints every statement with its bound values (emails, IDs, patient fields),
# so it is only on when ENVIRONMENT is development.
SQL_ECHO = is_development()

# Initialize asynchronous engine with connection pool
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=SQL_ECHO,
    pool_size=10,
    max_overflow=20,
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Administrative command-line utilities use a synchronous PostgreSQL session.
# The API continues to use AsyncSessionLocal and the asyncpg driver.
sync_engine = create_engine(
    make_url(settings.DATABASE_URL).set(drivername="postgresql+psycopg2"),
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=sync_engine, autoflush=False, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
