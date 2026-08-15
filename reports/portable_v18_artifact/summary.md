# Sales Sentinel V18 — Portable Runtime Artifact

- Artifact: **models/sales_sentinel_portable_v18.json.gz**
- SHA-256: `65e78860280ed9239f97176e9513f9c9803eede980355ac84983c07fe896fb7f`
- Training rows / features / trees: **541 / 96 / 1000**
- Compressed size: **0.47 MiB**
- Pure-Python parity max abs error: **4.441e-16**
- Static threshold: **0.242**
- Adaptive alert budget: **44.62%**

This artifact contains the exact trained tree structure and fold-local-style clipping metadata needed for inference without installing scikit-learn at runtime. RED remains disabled.
