from fastapi import APIRouter
from pathlib import Path
import joblib, sys, warnings, time
import pandas as pd

router   = APIRouter(prefix="/api", tags=["Sentiment"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Importar clases ANTES de joblib.load — orden crítico
sys.path.insert(0, str(Path(__file__).resolve().parent))
from migrar_pipeline_cpu import CpuFullPreprocessor, CpuPreprocessor, CpuTextCleaner
from inferencia_cpu import predecir_una_resena

# Registrar en __main__ para que pickle las encuentre
import __main__
__main__.CpuFullPreprocessor = CpuFullPreprocessor
__main__.CpuPreprocessor     = CpuPreprocessor
__main__.CpuTextCleaner      = CpuTextCleaner

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu_pipeline = joblib.load(BASE_DIR / "data/pipeline_transformacion_cpu.joblib")
        modelo_base  = joblib.load(BASE_DIR / "data/lightgbm_tuned_final_cpu.joblib")
        le_base      = joblib.load(BASE_DIR / "data/label_encoder.joblib")
    CLASSES = list(le_base.classes_)
    print(f"Modelos cargados. Clases: {CLASSES}")
except FileNotFoundError as e:
    print(f"Advertencia: {e}")
    cpu_pipeline = modelo_base = le_base = CLASSES = None


@router.post("/predict")
async def predict(body: dict):
    if not modelo_base or not cpu_pipeline:
        return {"error": "Modelos no cargados."}

    text  = str(body.get("text", ""))
    price = float(body.get("price", 0.0))

    try:
        t0     = time.time()
        result = predecir_una_resena(
            pipeline      = cpu_pipeline,
            modelo        = modelo_base,
            title         = text[:80],
            text          = text,
            price         = price,
            main_category = "Electronics",
            classes       = CLASSES,
        )
        latency = round((time.time() - t0) * 1000, 1)

        return {
            "label":         result["sentiment"],
            "probabilities": result["proba"],
            "model_used":    "baseline",
            "model_display": "Baseline (LightGBM + TF-IDF CPU)",
            "latency_ms":    latency,
        }
    except Exception as e:
        return {"error": f"Error en prediccion: {str(e)}"}