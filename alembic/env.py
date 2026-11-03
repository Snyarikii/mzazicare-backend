"""Migration environment for the existing MzaziCare PostgreSQL schema."""

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.base import Base
import app.models  # noqa: F401 - register models for autogenerate


target_metadata = Base.metadata


def _database_url() -> str:
    # The application uses asyncpg; Alembic runs migrations synchronously.
    # URL.__str__ masks the password, which makes the migration connection fail.
    return make_url(settings.DATABASE_URL).set(
        drivername="postgresql+psycopg2"
    ).render_as_string(hide_password=False)


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url(), poolclass=NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
