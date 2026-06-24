from fastapi import APIRouter
from pathlib import Path
import joblib
import pandas as pd

router  = APIRouter(prefix="/api", tags=["Sentiment"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent

try:
    cpu_pipeline = joblib.load(BASE_DIR / "data/pipeline_transformacion_cpu.joblib")
    modelo_base  = joblib.load(BASE_DIR / "data/lightgbm_tuned_final_cpu.joblib")
    le_base      = joblib.load(BASE_DIR / "data/label_encoder.joblib")
    print("Modelos de sentimiento cargados correctamente.")
except FileNotFoundError as e:
    print(f"Advertencia: No se encontraron los modelos. {e}")
    cpu_pipeline = modelo_base = le_base = None


@router.post("/predict")
async def predict(body: dict):
    if not modelo_base or not cpu_pipeline:
        return {"error": "Modelos no cargados en el servidor."}

    text     = str(body.get("text", ""))
    price    = float(body.get("price", 0.0))
    text_len = len(text)

    df_input = pd.DataFrame([{
        "text":     text,
        "text_len": text_len,
        "price":    price,
    }])

    try:
        X     = cpu_pipeline.transform(df_input)
        proba = modelo_base.predict_proba(X)[0]
        y_enc = modelo_base.predict(X)[0]
        label = le_base.classes_[int(y_enc)]

        return {
            "label":         label,
            "probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(le_base.classes_, proba)
            },
            "model_used":    "baseline",
            "model_display": "Baseline (LightGBM + TF-IDF CPU)",
            "latency_ms":    15,
        }
    except Exception as e:
        return {"error": f"Error en prediccion: {str(e)}"}