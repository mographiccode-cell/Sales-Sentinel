from __future__ import annotations

from datetime import timedelta
from pathlib import Path


def create_app(config_object=None):
    from flask import Flask, abort, redirect, render_template, request, session, url_for

    from .config import Config
    from .database import SessionLocal, init_engine
    from .services.i18n import date_value, locale, money, number, t
    from .services.security import csrf_token, current_user, validate_csrf

    config_object = config_object or Config
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_object)
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(
        seconds=app.config["PERMANENT_SESSION_LIFETIME_SECONDS"]
    )
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["REPORT_DIR"]).mkdir(parents=True, exist_ok=True)
    init_engine(app.config["DATABASE_URL"])

    from .admin.routes import admin_bp
    from .auth.routes import auth_bp
    from .dashboard.routes import dashboard_bp
    from .forecasting.routes import forecasting_bp
    from .imports.routes import imports_bp
    from .reports.routes import reports_bp
    from .sales.routes import sales_bp

    for blueprint in (auth_bp, dashboard_bp, sales_bp, forecasting_bp, imports_bp, reports_bp, admin_bp):
        app.register_blueprint(blueprint)

    @app.before_request
    def before_request():
        session.permanent = True
        if request.endpoint and request.endpoint != "static":
            validate_csrf()

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; font-src 'self' https://fonts.gstatic.com; "
            "style-src-elem 'self' 'unsafe-inline' https://fonts.googleapis.com",
        )
        return response

    @app.context_processor
    def context():
        user = current_user()
        return {
            "current_user": user,
            "t": t,
            "locale": locale(),
            "direction": "rtl" if locale() == "ar" else "ltr",
            "money": money,
            "number": number,
            "date_value": date_value,
            "csrf_token": csrf_token,
            "deployment_mode": app.config["DEPLOYMENT_MODE"],
        }

    @app.route("/locale/<language>")
    def set_locale(language):
        if language not in {"ar", "en"}:
            abort(404)
        session["locale"] = language
        next_url = request.args.get("next")
        if not next_url or not next_url.startswith("/") or next_url.startswith("//"):
            next_url = url_for("dashboard.index")
        return redirect(next_url)

    @app.get("/healthz")
    def healthz():
        from sqlalchemy import text
        from .database import session_scope
        try:
            with session_scope() as db:
                db.execute(text("SELECT 1"))
            return {"status": "ok", "database": "sqlite", "mode": app.config["DEPLOYMENT_MODE"]}, 200
        except Exception:
            return {"status": "error"}, 503

    @app.errorhandler(400)
    def error_400(error):
        return render_template("errors/error.html", code=400, message=str(getattr(error, "description", error))), 400

    @app.errorhandler(403)
    def error_403(_error):
        return render_template("errors/error.html", code=403, message="Access denied / غير مصرح بالوصول"), 403

    @app.errorhandler(404)
    def error_404(_error):
        return render_template("errors/error.html", code=404, message="Page not found / الصفحة غير موجودة"), 404

    @app.errorhandler(500)
    def error_500(_error):
        SessionLocal.remove()
        return render_template("errors/error.html", code=500, message="Unexpected system error / خطأ غير متوقع"), 500

    return app
