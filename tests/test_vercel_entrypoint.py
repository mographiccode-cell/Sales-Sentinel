from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_vercel_entrypoint_imports_and_serves_expected_public_and_protected_routes():
    repo = Path(__file__).resolve().parents[1]
    code = r'''
import json

import main

app = main.app
assert app is not None
assert app.config["DEPLOYMENT_MODE"] in {"vercel-demo-ephemeral", "vercel-demo-ephemeral-fallback"}
assert str(app.config["DATABASE_URL"]).startswith("sqlite:////tmp/")
assert str(app.config["UPLOAD_DIR"]).startswith("/tmp/")
assert str(app.config["REPORT_DIR"]).startswith("/tmp/")
assert app.config["SESSION_COOKIE_SECURE"] is True

rules = {rule.rule: rule.endpoint for rule in app.url_map.iter_rules()}
assert rules.get("/auth/login") == "auth.login", rules
assert rules.get("/") == "landing", rules
assert rules.get("/healthz") == "healthz", rules

client = app.test_client()
login = client.get("/auth/login")
assert login.status_code == 200, login.status_code
landing = client.get("/")
assert landing.status_code == 200, landing.status_code
health = client.get("/healthz")
assert health.status_code == 200, (health.status_code, health.data)
protected = client.get("/forecasts/", follow_redirects=False)
assert protected.status_code == 302, protected.status_code
location = protected.headers.get("Location", "")
assert "/auth/login" in location, location

print(json.dumps({
    "deployment_mode": app.config["DEPLOYMENT_MODE"],
    "database_url": app.config["DATABASE_URL"],
    "login_status": login.status_code,
    "landing_status": landing.status_code,
    "health_status": health.status_code,
    "protected_status": protected.status_code,
    "protected_location": location,
}))
'''
    env = os.environ.copy()
    env.update({
        "VERCEL": "1",
        "SECRET_KEY": "ci-vercel-entrypoint-secret",
    })
    env.pop("DATABASE_URL", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert completed.returncode == 0, f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    assert '"login_status": 200' in completed.stdout
    assert '"landing_status": 200' in completed.stdout
    assert '"health_status": 200' in completed.stdout
    assert '"protected_status": 302' in completed.stdout
