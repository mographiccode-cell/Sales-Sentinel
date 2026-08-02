from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from babel.dates import format_date as babel_format_date
from babel.numbers import format_currency, format_decimal
from flask import session

TRANSLATIONS = {
    "ar": {
        "app_name": "حارس المبيعات", "app_subtitle": "نظام التنبؤ الذكي بانخفاض المبيعات",
        "dashboard": "لوحة المعلومات", "forecasts": "التوقعات", "sales": "المبيعات",
        "alerts": "التنبيهات", "reports": "التقارير", "import_data": "استيراد البيانات",
        "imports": "استيراد البيانات", "users": "المستخدمون", "system_health": "حالة النظام",
        "logout": "تسجيل الخروج", "login": "تسجيل الدخول", "username": "اسم المستخدم",
        "password": "كلمة المرور", "current_sales": "المبيعات الحالية",
        "forecast_sales": "المبيعات المتوقعة", "decline_probability": "احتمال الانخفاض",
        "active_alerts": "التنبيهات النشطة",
        "demo_data_notice": "بيانات يومية حقيقية مجهولة الهوية من معرض إلكترونيات في جدة، يوليو–أكتوبر 2023. النسخة الحالية Pilot بسبب قصر الفترة.",
        "decision_support_notice": "التوقعات والتوصيات أدوات لدعم القرار وليست ضمانًا للنتائج.",
        "run_forecast": "تشغيل التنبؤ", "horizon": "الفترة المتوقعة", "days": "يومًا",
        "branch": "الفرع", "channel": "القناة", "all": "الكل", "model_quality": "جودة النموذج",
        "last_evaluation": "آخر تقييم", "data_size": "حجم البيانات", "lower_bound": "الحد الأدنى",
        "upper_bound": "الحد الأعلى", "baseline": "خط الأساس", "severity": "المستوى",
        "low": "منخفض", "medium": "متوسط", "high": "عالٍ", "critical": "حرج",
        "recommendations": "التوصيات", "factors": "العوامل المؤثرة", "search": "بحث",
        "download": "تنزيل", "print": "طباعة", "language": "English",
        "invalid_credentials": "اسم المستخدم أو كلمة المرور غير صحيحة.",
        "rate_limited": "تم تجاوز عدد محاولات الدخول. حاول لاحقًا.",
        "insufficient_data": "البيانات غير كافية للتنبؤ. يلزم توفر سجل زمني أطول.",
        "error": "حدث خطأ", "success": "نجحت العملية",
    },
    "en": {
        "app_name": "Sales Sentinel", "app_subtitle": "Intelligent Sales Decline Prediction System",
        "dashboard": "Dashboard", "forecasts": "Forecasts", "sales": "Sales", "alerts": "Alerts",
        "reports": "Reports", "import_data": "Import Data", "imports": "Import Data", "users": "Users",
        "system_health": "System Health", "logout": "Log out", "login": "Sign in", "username": "Username",
        "password": "Password", "current_sales": "Current sales", "forecast_sales": "Forecast sales",
        "decline_probability": "Decline probability", "active_alerts": "Active alerts",
        "demo_data_notice": "Real anonymized daily sales from a Jeddah electronics showroom, July–October 2023. This remains a pilot because the history is short.",
        "decision_support_notice": "Forecasts and recommendations support decisions; they do not guarantee outcomes.",
        "run_forecast": "Run forecast", "horizon": "Forecast horizon", "days": "days", "branch": "Branch",
        "channel": "Channel", "all": "All", "model_quality": "Model quality", "last_evaluation": "Last evaluation",
        "data_size": "Data size", "lower_bound": "Lower bound", "upper_bound": "Upper bound", "baseline": "Baseline",
        "severity": "Severity", "low": "Low", "medium": "Medium", "high": "High", "critical": "Critical",
        "recommendations": "Recommendations", "factors": "Contributing factors", "search": "Search",
        "download": "Download", "print": "Print", "language": "العربية",
        "invalid_credentials": "The username or password is incorrect.",
        "rate_limited": "Too many sign-in attempts. Try again later.",
        "insufficient_data": "There is not enough historical data to forecast reliably.",
        "error": "An error occurred", "success": "Operation completed",
    },
}


def locale() -> str:
    value = session.get("locale", "ar")
    return value if value in TRANSLATIONS else "ar"


def t(key: str) -> str:
    current = locale()
    return TRANSLATIONS.get(current, {}).get(key, TRANSLATIONS["en"].get(key, key))


def _number_locale() -> str:
    return "ar_SA" if locale() == "ar" else "en_US"


def money(value: float | Decimal | None) -> str:
    if value is None:
        return "—"
    return format_currency(value, "SAR", locale=_number_locale(), currency_digits=True)


def number(value: float | int | Decimal | None, digits: int = 0) -> str:
    if value is None:
        return "—"
    pattern = "#,##0" if digits == 0 else "#,##0." + ("0" * digits)
    return format_decimal(value, format=pattern, locale=_number_locale())


def date_value(value: date | datetime | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        value = value.date()
    return babel_format_date(value, format="medium", locale="ar_SA" if locale() == "ar" else "en_GB")
