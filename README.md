# Sales Sentinel

نظام Flask ثنائي اللغة لتحليل المبيعات اليومية والتنبؤ المبكر باحتمال الانخفاض. يستخدم SQLite وSQLAlchemy، ويحتفظ بملفات المصدر والتدريب والمودل والمقاييس بصورة قابلة للمراجعة وإعادة الإنتاج.

## البيانات والمودل

- المصدر: **UCI Online Retail**.
- DOI: `10.24432/C5BW33`.
- الترخيص: `CC BY 4.0`.
- السجلات الخام: `541,909`.
- السجلات النظيفة: `536,639` بعد إزالة `5,268` تكرارًا وصفّين بسعر سالب.
- الفترة: `2010-12-01` إلى `2011-12-09`.
- التقسيم: تدريب زمني، ثم 30 يومًا للتحقق، ثم 30 يومًا مستقلة للاختبار.
- المودل المختار: `ridge_raw_1` لأنه حقق أقل WAPE على مجموعة التحقق.
- نتيجة الاختبار المستقل: `WAPE = 23.63%`، و`MAE = 11,510.56`، و`RMSE = 15,830.77`.
- تغطية فاصل التوقع التجريبي: `86.67%`.

ملفات الإثبات:

- `scripts/build_uci_online_retail.py`
- `data/source_manifest.json`
- `data/processed/daily_sales.csv`
- `models/sales_forecast.json`
- `reports/model_metrics.json`
- `.github/workflows/academic-pipeline.yml`

## إعادة بناء البيانات وتدريب المودل

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-training.txt
python scripts/build_uci_online_retail.py
python -m pytest -q
```

يقوم Pipeline بتنزيل المصدر الرسمي، وتسجيل SHA-256 للملف المضغوط والملف الأصلي والملف المعالج، وتنظيف البيانات، ومقارنة Baselines ونماذج Ridge باستخدام تقسيم زمني، وحفظ المودل الأفضل ومقاييسه.

## التشغيل المحلي

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

يفتح التطبيق على `http://127.0.0.1:5000`، وتُبنى قاعدة SQLite تلقائيًا من البيانات اليومية الموثقة عند أول تشغيل.

## حسابات العرض

- `admin` / `Admin@2026!`
- `analyst` / `Analyst@2026!`

## النشر

يدعم Vercel عبر `vercel.json` و`api/index.py`. تستخدم نسخة Vercel SQLite مؤقتة داخل `/tmp` لأن نظام الملفات في الدوال Serverless مؤقت؛ التشغيل المحلي هو وضع SQLite الدائم للمشروع الأكاديمي.

## حدود الاستخدام

المودل أداة دعم قرار أكاديمية، وليس ضمانًا للنتائج المستقبلية. قد تختلف خصائص مؤسسة المستخدم عن متجر UCI، ولذلك يجب إعادة التدريب على بيانات المؤسسة عند توفرها. يدعم التطبيق توقعات 7 و30 يومًا فقط.
