from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import engine, get_db
from app.api.v1.endpoints import auth, caregiver, family, patients, telemetry
from app.services.risk_engine import risk_engine



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify database connectivity on application startup
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        print("Connected to PostgreSQL Docker container successfully")
    except Exception as exc:
        print(f"Database connection error on startup: {exc}")

    # Verify Risk engine model deployment
    if risk_engine.model is not None:
        print("[Risk Engine] Random Forest model is Loaded in memory and ready.", flush=True)
    else:
        print("[Risk Engine] Model artifact not found. Operating in heuristic mode.", flush=True)
    yield
    
    # Dispose connection pool on shutdown
    await engine.dispose()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Allow requests from mobile and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(f"{settings.API_V1_STR}/health", status_code=status.HTTP_200_OK)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Validates FastAPI application and PostgreSQL database responsiveness."""
    try:
        result = await db.execute(text("SELECT 1"))
        scalar_val = result.scalar()
        return {
            "status": "online",
            "database": "connected" if scalar_val == 1 else "unresponsive",
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
        }
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database probe failed: {str(err)}",
        )
    

app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(telemetry.router, prefix=settings.API_V1_STR)
app.include_router(patients.router, prefix=f"{settings.API_V1_STR}/patients", tags=["patients"])
app.include_router(family.router, prefix=settings.API_V1_STR)
app.include_router(caregiver.router, prefix=settings.API_V1_STR)
