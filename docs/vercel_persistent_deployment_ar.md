# مسار النشر الدائم لـ Sales Sentinel على Vercel

**الحالة التقنية:** جاهز من جهة الكود  
**قاعدة البيانات المحلية:** SQLite  
**قاعدة البيانات المقترحة للنشر الدائم:** PostgreSQL عبر `DATABASE_URL`

## لماذا لا نعتمد SQLite داخل /tmp على Vercel؟

عند تشغيل Flask داخل Vercel Functions لا يجب اعتبار `/tmp` قاعدة تخزين دائمة لبيانات المستخدم. النسخة الحالية تستطيع استخدام SQLite في `/tmp` كـDemo fallback، لكن البيانات المستوردة وسجل التوقعات والتنبيهات قد لا تبقى بعد تدوير الـFunction.

لذلك أصبح المشروع يعمل بوضعين واضحين:

### الوضع المحلي الأكاديمي

إذا لم يوجد `DATABASE_URL`:

- يستخدم SQLite تحت `instance/`.
- البيانات دائمة على الجهاز المحلي.
- لا يحتاج PostgreSQL.
- مناسب للعرض الأكاديمي والتشغيل المحلي.

### وضع النشر الدائم

إذا كان `DATABASE_URL` موجودًا:

- يستخدم PostgreSQL.
- لا يتم إنشاء SQLite `/tmp` كقاعدة أساسية.
- `/healthz` يعرض `database: postgresql`.
- `deployment_mode` يصبح `persistent-external-database`.
- معاملات CSV/XLSX والاستيراد والتوقعات والسجلات تحفظ في القاعدة الخارجية.

## التوافق الذي تم تنفيذه

تم تعديل طبقة قاعدة البيانات لتقبل الروابط الشائعة:

- `postgres://...`
- `postgresql://...`
- `postgresql+psycopg://...`

ويتم استخدام psycopg v3 داخل SQLAlchemy.

كما تم تعديل استيراد المبيعات ليستخدم:

- `INSERT OR IGNORE` عند SQLite.
- `ON CONFLICT (source_row_hash) DO NOTHING` عند PostgreSQL.

وبذلك تبقى حماية Duplicate Import فعالة في القاعدتين.

## إثبات PostgreSQL داخل CI

GitHub Actions يشغل حاوية **PostgreSQL 16** حقيقية ويختبر:

1. الاتصال بواسطة `DATABASE_URL`.
2. إنشاء جميع جداول SQLAlchemy.
3. ترقية `customer_key`.
4. استيراد معاملات فعلية.
5. حفظ Customer ID.
6. إعادة استيراد الملف نفسه.
7. التأكد من أن الصفوف لا تتكرر.
8. إنشاء تطبيق Flask كامل فوق PostgreSQL.
9. تشغيل `/healthz`.
10. التأكد من أن `/auth/login` يعمل.

آخر اختبار PostgreSQL الكامل انتهى بنجاح.

## Environment Variables المطلوبة

### DATABASE_URL

مثال عام:

```text
postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require
```

المشروع يقوم داخليًا بتحويل الرابط إلى psycopg v3 عند الحاجة.

### SECRET_KEY

يجب إنشاء قيمة عشوائية طويلة وعدم استخدام قيمة التطوير الافتراضية.

### SESSION_COOKIE_SECURE

في Vercel/HTTPS تكون مفعلة تلقائيًا، ويمكن ضبطها أيضًا على:

```text
1
```

## إنشاء PostgreSQL من Vercel

وفق توثيق Vercel الحالي يمكن ربط Storage Integration مثل Neon من Marketplace بالمشروع، ويقوم التكامل بتوفير متغيرات الاتصال للبيئات المختارة. يمكن كذلك استخدام أي PostgreSQL خارجي طالما يتم وضع رابط الاتصال في `DATABASE_URL`.

المشروع لا يعتمد على مزود PostgreSQL بعينه.

## ترتيب النشر الصحيح

1. Import مستودع `mographiccode-cell/Sales-Sentinel` كمشروع Vercel.
2. ربط PostgreSQL بالمشروع.
3. ضبط `DATABASE_URL`.
4. ضبط `SECRET_KEY`.
5. تنفيذ Production Deployment من `main`.
6. فتح `/healthz`.
7. يجب أن تكون النتيجة:

```json
{
  "status": "ok",
  "database": "postgresql",
  "mode": "persistent-external-database"
}
```

8. تسجيل الدخول.
9. رفع ملف Redsea أو ملف معاملات آخر بصيغة CSV/XLSX.
10. تنفيذ توقع 7 أيام والتحقق من ظهور V18 كمحرك Decline Risk.

## ماذا يحدث إذا لم يتم ربط PostgreSQL؟

يستمر الموقع في العمل على Vercel بوضع:

`vercel-demo-ephemeral`

وتظهر في صفحة الاستيراد رسالة تحذير واضحة بأن SQLite الموجودة في `/tmp` مؤقتة.

هذا مقصود حتى لا يتم إخفاء مشكلة استمرارية البيانات عن المستخدم.

## حالة V18 في النشر

V18 لا يحتاج scikit-learn في Runtime، لأنه يستخدم Artifact مضغوطًا وتنفيذ Pure Python:

- 96 Feature
- 1000 ExtraTrees
- ملف: `models/sales_sentinel_portable_v18.json.gz`
- محرك: `app/services/portable_decline_engine.py`

لذلك انتقال قاعدة البيانات من SQLite إلى PostgreSQL لا يغير المودل نفسه ولا يتطلب إعادة تدريبه.

## حدود هذا الملف

نجاح PostgreSQL في CI يثبت توافق الكود ومسار الاستيراد والتطبيق مع قاعدة دائمة. لكنه لا يعني أن Production Deployment على Vercel تم فعليًا؛ ذلك لا يحدث إلا بعد وجود Project مربوط في حساب Vercel وتنفيذ Deployment ناجح عليه.
