from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import DATABASE_URL

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise ValueError(
                "DATABASE_URL is not set. Use a Neon/Supabase URL, or for Colab "
                "without cloud DB set: sqlite:///fraud.db"
            )
        connect_args = {}
        if DATABASE_URL.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=not DATABASE_URL.startswith("sqlite"),
            connect_args=connect_args,
        )
    return _engine


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def reset_engine():
    """Clear cached engine (useful when DATABASE_URL changes in a notebook)."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
