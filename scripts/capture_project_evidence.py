from __future__ import annotations

import ast
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

from openpyxl import Workbook
from PIL import Image
from playwright.sync_api import Page, sync_playwright
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "screenshots"
INTERFACE = SCREENSHOTS / "Interface screens"
ERRORS = SCREENSHOTS / "Error screens"
FUNCTION_CODE = SCREENSHOTS / "Code function screenshots"
ERROR_CODE = SCREENSHOTS / "Code error screenshots"
BASE_URL = "http://127.0.0.1:5000"
VIEWPORT = {"width": 1440, "height": 900}


def reset_screenshot_directories() -> None:
    shutil.rmtree(SCREENSHOTS, ignore_errors=True)
    for folder in (INTERFACE, ERRORS, FUNCTION_CODE, ERROR_CODE):
        folder.mkdir(parents=True, exist_ok=True)


def build_seed_workbook(path: Path, days: int = 120) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet.append(["TRX DATE", "TRX NUMBER", "SALES CHANNEL", "CUSTOMER NUMBER", "ITEM CODE", "FAMILY", "CLASS", "SUBCLASS", "FRANCHISE", "Type", "QUANTITY", "Unit Price", "Discount Amount", "Discount Amount(%)", "Net Amount", "Vat Amount", "TOTAL AMOUNT"])
    start = date(2024, 1, 1)
    for day_index in range(days):
        recent_decline = day_index >= days - 21
        customer_count = 3 if recent_decline else 6
        level = 0.62 if recent_decline else 1.0
        for customer_index in range(customer_count):
            quantity = 1 + ((day_index + customer_index) % 4)
            channel = "Online" if customer_index % 3 == 0 else "Store"
            base_price = 95 + customer_index * 13 + (day_index % 7) * 3
            unit_price = round(base_price * level, 2)
            discount = 0.0 if customer_index % 4 else round(unit_price * 0.04, 2)
            net = round(quantity * unit_price - discount, 2)
            vat = round(net * 0.15, 2)
            total = round(net + vat, 2)
            sheet.append([start + timedelta(days=day_index), f"TRX-{day_index:03d}-{customer_index}", channel, f"CUST-{customer_index:03d}", f"ITEM-{customer_index % 5:02d}", "Retail", "General", f"Sub-{customer_index % 3}", "Saudi Demo", "INV", quantity, unit_price, discount, 4 if discount else 0, net, vat, total])
    workbook.save(path)
    workbook.close()


def screenshot(page: Page, path: Path) -> None:
    page.evaluate("window.scrollTo(0, 0)")
    page.screenshot(path=str(path), full_page=False)


def open_user_form(page: Page) -> None:
    if page.locator("#new-user-form").get_attribute("hidden") is not None:
        page.click('[data-toggle-target="#new-user-form"]')
    page.wait_for_selector("#new-user-form", state="visible")


def bypass_submit(page: Page, form_selector: str) -> None:
    page.locator(form_selector).evaluate("form => { form.noValidate = true; form.submit(); }")
    page.wait_for_load_state("networkidle")


def capture_browser_evidence() -> None:
    seed_path = Path("/tmp/sales sentinel evidence.xlsx")
    unsupported_path = Path("/tmp/unsupported notes.txt")
    build_seed_workbook(seed_path)
    unsupported_path.write_text("Unsupported evidence file", encoding="utf-8")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        page = context.new_page()
        page.goto(f"{BASE_URL}/auth/login", wait_until="networkidle")
        screenshot(page, INTERFACE / "1- Login screen.png")
        page.fill('input[name="username"]', "invalid user")
        page.fill('input[name="password"]', "wrong password")
        page.click('form.login-card button[type="submit"]')
        page.wait_for_selector(".flash.error", state="visible")
        screenshot(page, ERRORS / "1- Username or password test.png")
        page.fill('input[name="username"]', "admin")
        page.fill('input[name="password"]', "Admin@2026!")
        page.click('form.login-card button[type="submit"]')
        page.wait_for_url("**/dashboard")
        page.wait_for_load_state("networkidle")

        page.goto(f"{BASE_URL}/admin/users", wait_until="networkidle")
        open_user_form(page)
        page.fill('#new-user-form input[name="username"]', "weakuser")
        page.fill('#new-user-form input[name="email"]', "weak@example.com")
        page.fill('#new-user-form input[name="password"]', "123")
        page.fill('#new-user-form input[name="password_confirmation"]', "123")
        bypass_submit(page, "#new-user-form form")
        open_user_form(page)
        page.wait_for_selector(".flash.error", state="visible")
        screenshot(page, ERRORS / "2- Weak password test.png")

        page.goto(f"{BASE_URL}/admin/users", wait_until="networkidle")
        open_user_form(page)
        page.fill('#new-user-form input[name="username"]', "mismatchuser")
        page.fill('#new-user-form input[name="email"]', "mismatch@example.com")
        page.fill('#new-user-form input[name="password"]', "StrongPass123!")
        page.fill('#new-user-form input[name="password_confirmation"]', "DifferentPass123!")
        bypass_submit(page, "#new-user-form form")
        open_user_form(page)
        page.wait_for_selector(".flash.error", state="visible")
        screenshot(page, ERRORS / "3- Password confirmation test.png")

        page.goto(f"{BASE_URL}/imports/", wait_until="networkidle")
        bypass_submit(page, "form.ux3-upload-form")
        page.wait_for_selector(".flash.error", state="visible")
        screenshot(page, ERRORS / "4- Missing required field test.png")

        page.goto(f"{BASE_URL}/sales/", wait_until="networkidle")
        page.click('[data-toggle-target="#sales-filters"]')
        page.locator('#sales-filters input[name="start"]').evaluate("el => { el.type = 'text'; el.value = 'not-a-date'; }")
        page.fill('#sales-filters input[name="end"]', "2024-04-20")
        page.click('#sales-filters button[type="submit"]')
        page.wait_for_load_state("networkidle")
        page.click('[data-toggle-target="#sales-filters"]')
        page.wait_for_selector(".flash.error", state="visible")
        screenshot(page, ERRORS / "5- Invalid date test.png")

        page.goto(f"{BASE_URL}/imports/", wait_until="networkidle")
        page.set_input_files('form.ux3-upload-form input[name="file"]', str(unsupported_path))
        page.click('form.ux3-upload-form button[type="submit"]')
        page.wait_for_load_state("networkidle")
        page.wait_for_selector(".flash.error", state="visible")
        screenshot(page, ERRORS / "6- Unsupported file test.png")

        page.goto(f"{BASE_URL}/imports/", wait_until="networkidle")
        screenshot(page, INTERFACE / "3- New analysis interface.png")
        page.set_input_files('form.ux3-upload-form input[name="file"]', str(seed_path))
        page.click('form.ux3-upload-form button[type="submit"]')
        page.wait_for_selector("#instant-analysis", timeout=60000)
        screenshot(page, INTERFACE / "4- Analysis result interface.png")
        page.goto(f"{BASE_URL}/dashboard", wait_until="networkidle")
        screenshot(page, INTERFACE / "2- User dashboard.png")
        page.goto(f"{BASE_URL}/forecasts/", wait_until="networkidle")
        page.wait_for_selector("tr[data-row-href]", state="visible")
        screenshot(page, INTERFACE / "5- Analysis history screen.png")
        run_href = page.locator("tr[data-row-href]").first.get_attribute("data-row-href")
        if not run_href:
            raise RuntimeError("No analysis detail route was created after real browser upload")
        page.goto(f"{BASE_URL}{run_href}", wait_until="networkidle")
        screenshot(page, INTERFACE / "6- Forecast overview screen.png")
        page.click('[data-ux-tab="forecast"]')
        page.wait_for_selector('[data-ux-panel="forecast"]', state="visible")
        screenshot(page, INTERFACE / "7- Daily forecast interface.png")
        page.click('[data-ux-tab="quality"]')
        page.wait_for_selector('[data-ux-panel="quality"]', state="visible")
        screenshot(page, INTERFACE / "8- Model quality screen.png")

        page.goto(f"{BASE_URL}/sales/", wait_until="networkidle")
        page.click('[data-toggle-target="#sales-filters"]')
        page.fill('#sales-filters input[name="start"]', "2024-04-01")
        page.fill('#sales-filters input[name="end"]', "2024-04-29")
        page.select_option('#sales-filters select[name="channel"]', label="Online")
        page.click('#sales-filters button[type="submit"]')
        page.wait_for_load_state("networkidle")
        page.click('[data-toggle-target="#sales-filters"]')
        screenshot(page, INTERFACE / "9- Sales search screen.png")
        page.goto(f"{BASE_URL}/reports/", wait_until="networkidle")
        screenshot(page, INTERFACE / "10- Reports interface.png")

        page.goto(f"{BASE_URL}/admin/users", wait_until="networkidle")
        open_user_form(page)
        page.fill('#new-user-form input[name="username"]', "reviewanalyst")
        page.fill('#new-user-form input[name="email"]', "reviewanalyst@example.com")
        page.fill('#new-user-form input[name="password"]', "ReviewPass123!")
        page.fill('#new-user-form input[name="password_confirmation"]', "ReviewPass123!")
        page.click('#new-user-form button[type="submit"]')
        page.wait_for_load_state("networkidle")
        page.goto(f"{BASE_URL}/admin/users", wait_until="networkidle")
        if page.locator("details").count() > 1:
            page.locator("details").nth(1).evaluate("el => el.open = true")
        screenshot(page, INTERFACE / "11- User management interface.png")
        page.goto(f"{BASE_URL}/admin/roles", wait_until="networkidle")
        screenshot(page, INTERFACE / "12- Roles and permissions interface.png")
        page.goto(f"{BASE_URL}/admin/health", wait_until="networkidle")
        screenshot(page, INTERFACE / "13- System health screen.png")
        page.goto(f"{BASE_URL}/admin/settings", wait_until="networkidle")
        screenshot(page, INTERFACE / "14- Settings interface.png")
        page.goto(f"{BASE_URL}/alerts/", wait_until="networkidle")
        screenshot(page, INTERFACE / "15- Alert center interface.png")
        context.close()
        browser.close()


def function_range(path: Path, function_name: str, max_lines: int) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            start = min([node.lineno] + [item.lineno for item in node.decorator_list])
            end = min(node.end_lineno or start, start + max_lines - 1)
            return start, end
    raise ValueError(f"Function {function_name!r} not found in {path}")


def needle_range(path: Path, needle: str, *, before: int, after: int) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, start=1):
        if needle in line:
            return max(1, index - before), min(len(lines), index + after)
    raise ValueError(f"Needle {needle!r} not found in {path}")


def code_html(snippet: str, start_line: int) -> str:
    formatter = HtmlFormatter(style="monokai", linenos="table", linenostart=start_line, noclasses=True)
    rendered = highlight(snippet, PythonLexer(), formatter)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>*{{box-sizing:border-box}}html,body{{margin:0;padding:0;background:#1e1e1e;color:#d4d4d4}}body{{display:inline-block;min-width:1180px;max-width:1400px}}.frame{{display:inline-block;min-width:1180px;max-width:1400px;background:#1e1e1e;padding:18px 22px 18px 12px}}.highlighttable{{border-spacing:0;border-collapse:collapse;margin:0;width:100%}}.linenos{{width:64px;min-width:64px;padding:0 16px 0 4px!important;border-right:1px solid #343434;vertical-align:top;user-select:none}}.linenos pre{{color:#858585!important;text-align:right!important;background:transparent!important}}.code{{padding-left:18px!important;vertical-align:top;width:100%}}pre{{margin:0!important;font-family:'Cascadia Code','DejaVu Sans Mono','Liberation Mono',monospace!important;font-size:15px!important;line-height:1.48!important;tab-size:4;white-space:pre-wrap!important;overflow-wrap:normal;word-break:normal;direction:ltr;text-align:left}}</style></head><body><div class="frame">{rendered}</div></body></html>"""


def render_code_image(page: Page, path: Path, start: int, end: int, output: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    snippet = "\n".join(lines[start - 1 : end]) + "\n"
    page.set_content(code_html(snippet, start), wait_until="load")
    frame = page.locator(".frame")
    box = frame.bounding_box()
    if not box:
        raise RuntimeError(f"Could not measure code frame for {output.name}")
    height = min(990, int(box["height"] + 1))
    width = min(1400, max(1180, int(box["width"] + 1)))
    page.set_viewport_size({"width": width, "height": height})
    page.screenshot(path=str(output), clip={"x": 0, "y": 0, "width": width, "height": height})


def capture_code_evidence() -> None:
    functions = [
        ("1- User login function.png", "app/auth/routes.py", "login", 39),
        ("2- User creation function.png", "app/admin/routes.py", "create_user", 40),
        ("3- Dashboard summary function.png", "app/dashboard/routes.py", "index", 28),
        ("4- Sales search function.png", "app/sales/routes.py", "index", 40),
        ("5- Sales import function.png", "app/imports/routes.py", "index", 42),
        ("6- Instant analysis function.png", "app/services/instant_analysis.py", "run_instant_analysis", 42),
        ("7- Decline explanation function.png", "app/services/decline_explainer.py", "explain_decline_drivers", 42),
        ("8- Report generation function.png", "app/reports/routes.py", "_pdf_payload", 42),
    ]
    error_blocks = [
        ("1- Username or password test code.png", "app/auth/routes.py", "or not verify_password", 4, 17),
        ("2- Weak password test code.png", "app/admin/routes.py", "if len(password) < 10:", 2, 13),
        ("3- Password confirmation test code.png", "app/admin/routes.py", "if password != password_confirmation:", 2, 13),
        ("4- Missing required field test code.png", "app/imports/routes.py", "if not uploaded or not uploaded.filename:", 2, 10),
        ("5- Invalid date test code.png", "app/sales/routes.py", "except ValueError:", 4, 17),
        ("6- Unsupported file test code.png", "app/imports/routes.py", 'if extension not in {".csv", ".xlsx"}:', 2, 12),
    ]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        for name, relative, function_name, max_lines in functions:
            source = ROOT / relative
            start, end = function_range(source, function_name, max_lines)
            render_code_image(page, source, start, end, FUNCTION_CODE / name)
        for name, relative, needle, before, after in error_blocks:
            source = ROOT / relative
            start, end = needle_range(source, needle, before=before, after=after)
            render_code_image(page, source, start, end, ERROR_CODE / name)
        browser.close()


def verify_output() -> None:
    expected_directories = {"Interface screens", "Error screens", "Code function screenshots", "Code error screenshots"}
    actual_directories = {path.name for path in SCREENSHOTS.iterdir() if path.is_dir()}
    if actual_directories != expected_directories:
        raise AssertionError(f"Unexpected screenshot directories: {sorted(actual_directories)}")
    expected_counts = {INTERFACE: 15, ERRORS: 6, FUNCTION_CODE: 8, ERROR_CODE: 6}
    manifest: list[dict[str, object]] = []
    for folder, expected_count in expected_counts.items():
        files = sorted(folder.glob("*.png"))
        if len(files) != expected_count:
            raise AssertionError(f"{folder.name} has {len(files)} PNG files; expected {expected_count}")
        for file in files:
            if "_" in file.name or "_" in folder.name:
                raise AssertionError(f"Underscore is not allowed: {file}")
            with Image.open(file) as image:
                image.verify()
            with Image.open(file) as image:
                width, height = image.size
            if height > 1000:
                raise AssertionError(f"Image exceeds 1000px height: {file} -> {height}")
            manifest.append({"name": file.name, "category": folder.name, "width": width, "height": height})
    error_names = {path.name.replace(".png", "") for path in ERRORS.glob("*.png")}
    error_code_names = {path.name.replace(" code.png", "") for path in ERROR_CODE.glob("*.png")}
    if error_names != error_code_names:
        raise AssertionError("Error screen names and error-code screenshot names do not match: " f"screens={sorted(error_names)} code={sorted(error_code_names)}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    reset_screenshot_directories()
    capture_browser_evidence()
    capture_code_evidence()
    verify_output()


if __name__ == "__main__":
    main()
