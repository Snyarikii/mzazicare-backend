import asyncio
from sqlalchemy.dialects.postgresql import insert
from app.db.base import Base
from app.db.session import engine
from app.models.entities import MockCaregiverRegistry
import app.models  # noqa: F401 - registers all entities to Base.metadata


async def init_models():
  async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)
    await conn.execute(
      insert(MockCaregiverRegistry).values([
        {"license_number": "CG-2026-001", "full_name": "Stephen Nyariki"},
        {"license_number": "CG-2026-002", "full_name": "Lydia Naliaka"},
        {"license_number": "CG-2026-003", "full_name": "Mercy Atieno"},
      ]).on_conflict_do_nothing(index_elements=["license_number"])
    )
  await engine.dispose()
  print("All database tables created successfully in PostgreSQL!")


if __name__ == "__main__":
  asyncio.run(init_models())
