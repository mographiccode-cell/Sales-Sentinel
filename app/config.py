from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IS_VERCEL = bool(os.getenv("VERCEL"))
EXTERNAL_DATABASE_URL = str(os.getenv("DATABASE_URL") or "").strip()


def prepare_runtime_database() -> Path:
    """Return the SQLite path used when no external database is configured.

    Local development uses persistent SQLite under ``instance``. Vercel falls
    back to an ephemeral ``/tmp`` SQLite copy only when ``DATABASE_URL`` is not
    configured. A persistent hosted database should therefore be supplied for
    deployments that need imports, forecasts, alerts, and audit history to
    survive function recycling.
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
    DATABASE_PATH = None if EXTERNAL_DATABASE_URL else prepare_runtime_database()
    DATABASE_URL = EXTERNAL_DATABASE_URL or f"sqlite:///{DATABASE_PATH}"
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
    if EXTERNAL_DATABASE_URL:
        DEPLOYMENT_MODE = "persistent-external-database"
    elif IS_VERCEL:
        DEPLOYMENT_MODE = "vercel-demo-ephemeral"
    else:
        DEPLOYMENT_MODE = "local-persistent-sqlite"
