from __future__ import annotations

from flask import request

from app.models import AuditLog


def write_audit(db, action: str, user_id: int | None = None, entity_type: str | None = None,
                entity_id: str | None = None, details: dict | None = None) -> AuditLog:
    log = AuditLog(
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        ip_address=(request.headers.get("X-Forwarded-For", request.remote_addr) or "")[:64] if request else None,
        user_agent=(request.user_agent.string or "")[:255] if request else None,
        details_json=details or {},
    )
    db.add(log)
    return log
