"""
delta-chat · scripts/setup_db.py
Initialise database schema. Run once before first use.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observability.logging import configure_logging

configure_logging()

from src.db.session import init_db, health_check
from src.observability.logging import get_logger

log = get_logger(__name__)


def main():
    print("Initialising database schema...")
    try:
        init_db()
        ok = health_check()
        if ok:
            print("[OK] Database initialised and healthy.")
        else:
            print("[FAIL] Database initialised but health check failed. Check DATABASE_URL.")
            sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Database initialisation failed: {e}")
        print("  Check DATABASE_URL in .env")
        print("  For local dev without MySQL: DATABASE_URL=sqlite:///./delta_chat.db")
        sys.exit(1)


if __name__ == "__main__":
    main()
