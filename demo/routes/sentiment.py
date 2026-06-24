from fastapi import APIRouter
from pathlib import Path
import joblib, sys, warnings, time
import lightgbm  # debe cargarse antes de torch en macOS para evitar conflicto OpenMP
import pandas as pd
import traceback
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from peft import PeftModel

router   = APIRouter(prefix="/api", tags=["Sentiment"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Importar clases ANTES de joblib.load
sys.path.insert(0, str(Path(__file__).resolve().parent))
from migrar_pipeline_cpu import CpuFullPreprocessor, CpuPreprocessor, CpuTextCleaner
from inferencia_cpu import predecir_una_resena

import __main__
__main__.CpuFullPreprocessor = CpuFullPreprocessor
__main__.CpuPreprocessor     = CpuPreprocessor
__main__.CpuTextCleaner      = CpuTextCleaner

# ── BASELINE ────────────────────────────────────────
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu_pipeline = joblib.load(BASE_DIR / "data/pipeline_transformacion_cpu.joblib")
        modelo_base  = joblib.load(BASE_DIR / "data/lightgbm_tuned_final_cpu.joblib")
        le_base      = joblib.load(BASE_DIR / "data/label_encoder.joblib")
    CLASSES = [str(c) for c in le_base.classes_]
    print(f"Modelos cargados. Clases: {CLASSES}")
except FileNotFoundError as e:
    print(f"Advertencia: {e}")
    cpu_pipeline = modelo_base = le_base = CLASSES = None

# ── TRANSFORMER — carga al arrancar ─────────────────
try:
    _lora_path = BASE_DIR / "models/distilbert_lora"
    _db_device = "cuda" if torch.cuda.is_available() else "cpu"
    _db_tok    = DistilBertTokenizerFast.from_pretrained(str(_lora_path))
    _base      = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3, ignore_mismatched_sizes=True
    )
    _db_model  = PeftModel.from_pretrained(_base, str(_lora_path)).to(_db_device)
    _db_model.eval()
    print(f"DistilBERT cargado en {_db_device}")
except Exception as e:
    print(f"DistilBERT no disponible: {e}")
    _db_model = _db_tok = None
    _db_device = "cpu"


# ── ENDPOINT ─────────────────────────────────────────
@router.post("/predict")
def predict(body: dict):
    model = body.get("model", "baseline")
    text  = str(body.get("text", ""))
    print(f"Request recibido: model={model}, text_len={len(text)}")

    try:
        raw_price = body.get("price", 0.0)
        price = float(raw_price) if raw_price is not None else 0.0
    except (ValueError, TypeError):
        price = 0.0

    if not text.strip():
        return {"error": "Texto vacío."}

    # ── BASELINE ──────────────────────────────────────
    if model == "baseline":
        if not modelo_base or not cpu_pipeline or not CLASSES:
            return {"error": "Modelos no cargados."}
        try:
            t0 = time.time()
            df = pd.DataFrame([{
                "title":         text[:80],
                "text":          text,
                "text_len":      len(text),
                "price":         price,
                "main_category": "Electronics",
            }])
            X     = cpu_pipeline.transform(df)
            proba = modelo_base.predict_proba(X)[0]
            y_enc = modelo_base.predict(X)[0]
            label = CLASSES[int(y_enc)]
            return {
                "label":         label,
                "probabilities": {cls: round(float(p), 4) for cls, p in zip(CLASSES, proba)},
                "model_used":    "baseline",
                "model_display": "Baseline (LightGBM + TF-IDF CPU)",
                "latency_ms":    round((time.time() - t0) * 1000, 1),
            }
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error en prediccion: {str(e)}"}

    # ── TRANSFORMER ───────────────────────────────────
    if model == "transformer":
        try:
            if not CLASSES:
                return {"error": "Clases no cargadas."}
            if _db_model is None or _db_tok is None:
                return {"error": "Transformer no disponible. Revisa que los archivos de distilbert_lora esten en models/."}

            t0  = time.time()
            enc = _db_tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            enc = {k: v.to(_db_device) for k, v in enc.items()}
            with torch.no_grad():
                logits = _db_model(**enc).logits
            proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            y_enc = int(proba.argmax())
            label = CLASSES[y_enc]

            return {
                "label":         label,
                "probabilities": {cls: round(float(p), 4) for cls, p in zip(CLASSES, proba)},
                "model_used":    "transformer",
                "model_display": "Transformer (DistilBERT + LoRA)",
                "latency_ms":    round((time.time() - t0) * 1000, 1),
            }
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error en transformer: {str(e)}"}

    return {"error": f"Modelo desconocido: {model}"}