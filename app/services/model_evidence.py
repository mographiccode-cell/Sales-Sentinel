from __future__ import annotations

from app.services.portable_decline_engine import load_artifact


def enrich_analysis_evidence(analysis: dict) -> dict:
    """Attach only metrics that are actually embedded in the deployed V18 artifact.

    The preferred evidence block is the calibrated Redsea post-open diagnostic.
    It is explicitly labelled diagnostic (not fresh validation) to avoid turning
    historical evidence into a misleading accuracy claim for a new upload.
    """
    if not analysis or not analysis.get("available"):
        return analysis

    artifact = load_artifact()
    evidence = artifact.get("evidence") or {}
    preferred_key = "v16_2_calibrated_redsea_post_open"
    block = evidence.get(preferred_key)
    if not isinstance(block, dict):
        candidates = [(key, value) for key, value in evidence.items() if isinstance(value, dict) and value.get("accuracy") is not None]
        if candidates:
            preferred_key, block = candidates[-1]
        else:
            block = {}

    accuracy = block.get("accuracy")
    tp = int(block.get("tp") or 0)
    tn = int(block.get("tn") or 0)
    fp = int(block.get("fp") or 0)
    fn = int(block.get("fn") or 0)
    sample_size = tp + tn + fp + fn

    analysis.update({
        "decline_diagnostic_accuracy_pct": float(accuracy) * 100.0 if accuracy is not None else None,
        "decline_diagnostic_error_pct": (1.0 - float(accuracy)) * 100.0 if accuracy is not None else None,
        "decline_correct_count": tp + tn if sample_size else None,
        "decline_wrong_count": fp + fn if sample_size else None,
        "decline_diagnostic_sample_size": sample_size or None,
        "decline_precision_pct": float(block.get("precision")) * 100.0 if block.get("precision") is not None else None,
        "decline_recall_pct": float(block.get("recall")) * 100.0 if block.get("recall") is not None else None,
        "decline_f1_pct": float(block.get("f1")) * 100.0 if block.get("f1") is not None else None,
        "decline_roc_auc_pct": float(block.get("roc_auc")) * 100.0 if block.get("roc_auc") is not None else None,
        "decline_evidence_label": preferred_key,
        "scientific_status": artifact.get("scientific_status"),
        "decline_target_definition": artifact.get("target_definition"),
    })
    return analysis
