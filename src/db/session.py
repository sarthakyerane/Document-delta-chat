"""
delta-chat · src/db/session.py
SQLAlchemy engine + session factory.
Supports MySQL (production) and SQLite (local dev / CI).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.db.models import Base
from src.observability.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_engine():
    settings = get_settings()
    url = settings.database_url
    kwargs: dict = {}
    if url.startswith("mysql"):
        kwargs = {
            "pool_pre_ping": True,
            "pool_size": 5,
            "max_overflow": 10,
            "pool_recycle": 3600,
        }
    elif url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
    engine = create_engine(url, **kwargs)
    log.info("db.engine.created", url=url.split("@")[-1])  # don't log credentials
    return engine


def get_engine():
    return _get_engine()


def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a DB session and closes it after use."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Call at startup."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    log.info("db.init.complete")


def health_check() -> bool:
    """Verify DB connectivity."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.error("db.health_check.failed", error=str(e))
        return False
