from __future__ import annotations

import hashlib
import traceback

try:
    from app import create_app

    app = create_app()
except Exception as startup_error:  # pragma: no cover - Vercel safety net
    # Never let an import/startup exception become Vercel's opaque
    # FUNCTION_INVOCATION_FAILED page. Expose only sanitized diagnostics so we
    # can identify the failing component without leaking secrets or URLs.
    from flask import Flask, jsonify

    app = Flask(__name__)

    error_type = type(startup_error).__name__
    error_text = str(startup_error)
    error_fingerprint = hashlib.sha256(
        f"{error_type}:{error_text}".encode("utf-8", errors="replace")
    ).hexdigest()[:16]

    tb = traceback.extract_tb(startup_error.__traceback__)
    last_frame = tb[-1] if tb else None
    safe_location = (
        f"{last_frame.name}:{last_frame.lineno}" if last_frame else "startup"
    )

    @app.get("/")
    def startup_failure_page():
        return (
            "Sales Sentinel is temporarily starting in recovery mode. "
            "Check /healthz for the sanitized startup diagnostic.",
            503,
            {"Content-Type": "text/plain; charset=utf-8"},
        )

    @app.get("/healthz")
    def startup_health():
        return jsonify(
            {
                "status": "startup_error",
                "error_type": error_type,
                "error_fingerprint": error_fingerprint,
                "location": safe_location,
            }
        ), 503
