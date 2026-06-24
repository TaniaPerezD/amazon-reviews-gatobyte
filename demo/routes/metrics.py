# Agregar a main.py o a un router nuevo demo/routes/metrics.py

from fastapi import APIRouter
from pathlib import Path
import json

router   = APIRouter(prefix="/api", tags=["Metrics"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent

@router.get("/metrics")
async def get_metrics():
    result = {}

    # ── Baseline
    baseline_path = BASE_DIR / "data/metadata_modelo_final.json"
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            b = json.load(f)
        result["baseline"] = {
            "available":     True,
            "model_display": "Baseline",
            "model_detail":  f"LightGBM + TF-IDF ({b.get('tfidf_max_features', 10000)} features)",
            "metricas_val":  b.get("metricas_val"),
            "metricas_test": b.get("metricas_test"),
            "clases":        b.get("clases", []),
        }
    else:
        result["baseline"] = {"available": False}

    # ── Transformer
    trans_path = BASE_DIR / "data/metadata_transformer.json"
    if trans_path.exists():
        with open(trans_path, "r") as f:
            t = json.load(f)
        # Si las metricas son null, marcar como no disponible aun
        test_metrics = t.get("metricas_test", {})
        has_metrics  = test_metrics and test_metrics.get("f1_macro") is not None
        result["transformer"] = {
            "available":     has_metrics,
            "model_display": "Transformer",
            "model_detail":  "DistilBERT + LoRA (PEFT)",
            "metricas_val":  t.get("metricas_val")  if has_metrics else None,
            "metricas_test": t.get("metricas_test") if has_metrics else None,
            "clases":        t.get("clases", []),
        }
    else:
        result["transformer"] = {"available": False}

    return result