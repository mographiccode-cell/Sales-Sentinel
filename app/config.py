from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IS_VERCEL = bool(os.getenv("VERCEL"))


def prepare_runtime_database() -> Path:
    """Return a writable SQLite path.

    Locally the project uses the persistent database under ``instance``.
    Vercel Functions have an ephemeral read-only deployment filesystem, so the
    verified seed database is copied to ``/tmp`` for the lifetime of a warm
    function instance. The public deployment is therefore a demonstration;
    local SQLite remains the authoritative persistent mode.
    """
    seed = BASE_DIR / "instance" / "sales_sentinel.db"
    if not IS_VERCEL:
        seed.parent.mkdir(parents=True, exist_ok=True)
        return seed
    target = Path("/tmp/sales_sentinel.db")
    if not target.exists():
        if seed.exists():
            shutil.copy2(seed, target)
        else:
            bootstrap = BASE_DIR / "data" / "bootstrap.sql"
            if bootstrap.exists():
                connection = sqlite3.connect(target)
                try:
                    connection.executescript(bootstrap.read_text(encoding="utf-8"))
                    connection.commit()
                finally:
                    connection.close()
    return target


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "development-only-change-before-production-2026")
    DATABASE_PATH = prepare_runtime_database()
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")
    UPLOAD_DIR = Path("/tmp/uploads") if IS_VERCEL else BASE_DIR / "instance" / "uploads"
    REPORT_DIR = Path("/tmp/reports") if IS_VERCEL else BASE_DIR / "instance" / "reports"
    MODEL_DIR = BASE_DIR / "models"
    FORECAST_MODEL_PATH = MODEL_DIR / "moving_average_v1.json"
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    PERMANENT_SESSION_LIFETIME_SECONDS = 3600
    DECLINE_THRESHOLD = float(os.getenv("DECLINE_THRESHOLD", "0.08"))
    MIN_HISTORY_DAYS = 90
    TIMEZONE = "Asia/Riyadh"
    LOGIN_RATE_LIMIT = 5
    LOGIN_RATE_WINDOW_SECONDS = 900
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = IS_VERCEL or os.getenv("SESSION_COOKIE_SECURE", "0") == "1"
    DEPLOYMENT_MODE = "vercel-demo-ephemeral" if IS_VERCEL else "local-persistent-sqlite"
