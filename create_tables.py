import asyncio
from app.db.base import Base
from app.db.session import engine
import app.models  # noqa: F401 - registers all entities to Base.metadata


async def init_models():
  async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)
  await engine.dispose()
  print("All database tables created successfully in PostgreSQL!")


if __name__ == "__main__":
  asyncio.run(init_models())