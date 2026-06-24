# demo/routes/rag.py
import json
import pickle
import time
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
import yaml
from fastapi import APIRouter, Query # 1. Importamos APIRouter en lugar de FastAPI
from sentence_transformers import SentenceTransformer

# Como este archivo está dentro de demo/routes/, subimos 3 niveles para la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 2. Creamos el router. Le ponemos el prefijo "/api" para no tener que escribirlo en cada ruta.
router = APIRouter(prefix="/api", tags=["RAG"])

# ─────────────────────────────────────────────────────────
# Carga de recursos (Se queda exactamente igual)
# ─────────────────────────────────────────────────────────
def load_config():
    with open(BASE_DIR / "config" / "rag_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_resources():
    cfg = load_config()
    index_dir = BASE_DIR / cfg["faiss"]["index_dir"]
    latest = index_dir / "latest.json"

    if not latest.exists():
        raise FileNotFoundError(
            "Índice no encontrado. Ejecuta primero: python src/01_build_index.py"
        )

    with open(latest, "r", encoding="utf-8") as f:
        ptr = json.load(f)

    with open(ptr["manifest_path"], "r", encoding="utf-8") as f:
        manifest = json.load(f)

    index = faiss.read_index(manifest["index_file"])

    with open(manifest["metadata_file"], "rb") as f:
        chunks = pickle.load(f)

    model = SentenceTransformer(cfg["embedding"]["model_name"])

    return cfg, index, chunks, manifest, model

print("Cargando recursos RAG...")
try:
    cfg, index, chunks, manifest, model = load_resources()
    print(f"✅ Listo — {manifest['num_chunks']:,} fragmentos indexados")
except FileNotFoundError as e:
    print(f"⚠️  Índice no encontrado: {e}")
    print("   Corre: python src/01_build_index.py")
    cfg = load_config()
    index = chunks = manifest = model = None


# ─────────────────────────────────────────────────────────
# Rutas usando 'router' (Quitamos el '/api' del string porque ya está en el prefix)
# ─────────────────────────────────────────────────────────

@router.get("/info")  # Antes era @app.get("/api/info")
async def get_info():
    """Información del índice activo."""
    created_str = manifest.get("created_at_utc", "")
    age_days = 0
    if created_str:
        created_dt = datetime.datetime.strptime(created_str, "%Y%m%d_%H%M%S")
        age_days = (datetime.datetime.utcnow() - created_dt).total_seconds() / 86400

    policy = cfg["update_policy"]
    age_ok = age_days <= policy["max_age_days"]

    return {
        "num_chunks": manifest["num_chunks"],
        "sample_size": manifest["sample_size"],
        "embedding_dim": manifest["embedding_dim"],
        "model_name": manifest["model_name"],
        "version": manifest["version"][-15:],
        "age_days": round(age_days, 1),
        "status": "OK" if age_ok else "TRIGGER",
        "keep_n_versions": cfg["faiss"]["keep_n_versions"],
        "chunk_overlap": manifest.get("chunk_overlap", "N/A"),
        "default_top_k": cfg["retrieval"]["default_top_k"],
    }


@router.get("/search")  # Antes era @app.get("/api/search")
async def search(
    q: str = Query(..., min_length=1, description="Consulta de búsqueda"),
    top_k: int = Query(5, ge=1, le=20),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
):
    """Búsqueda semántica sobre el índice FAISS."""
    t0 = time.time()

    q_vec = model.encode(
        [q],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(q_vec, top_k * 3)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or float(score) < min_score:
            continue
        c = chunks[idx]
        results.append({
            "score": round(float(score), 4),
            "chunk_text": c.get("chunk_text", ""),
            "chunk_number": c.get("chunk_number", 0),
            "rating": c.get("rating"),
            "helpful_vote": c.get("helpful_vote", 0),
            "parent_asin": c.get("parent_asin", ""),
            "title": c.get("title", "") or f"Producto {c.get('parent_asin', '')}",
            "sentiment": c.get("sentiment", ""),
            "main_category": c.get("main_category", "Electrónica"),
        })
        if len(results) == top_k:
            break

    latency_ms = round((time.time() - t0) * 1000, 1)
    avg_score = round(float(np.mean([r["score"] for r in results])), 4) if results else 0
    avg_rating = None
    ratings = [float(r["rating"]) for r in results if r["rating"] is not None]
    if ratings:
        avg_rating = round(float(np.mean(ratings)), 2)

    return {
        "query": q,
        "results": results,
        "meta": {
            "total": len(results),
            "latency_ms": latency_ms,
            "avg_score": avg_score,
            "avg_rating": avg_rating,
            "top_k": top_k,
        }
    }


@router.get("/chunks")  # Antes era @app.get("/api/chunks")
async def get_chunks(
    sentiment: Optional[str] = None,
    category: Optional[str] = None,
    n: int = Query(50, ge=1, le=500),
):
    """Explorador de chunks con filtros."""
    filtered = [
        c for c in chunks
        if (not sentiment or sentiment == "Todos" or c.get("sentiment") == sentiment)
        and (not category or category == "Todos" or c.get("main_category") == category)
    ][:n]

    rows = []
    for c in filtered:
        rows.append({
            "chunk": c.get("chunk_number", 0),
            "asin": c.get("parent_asin", ""),
            "sentiment": c.get("sentiment", ""),
            "category": c.get("main_category", ""),
            "rating": c.get("rating"),
            "text": (c.get("chunk_text", "") or "")[:220],
        })

    return {"rows": rows, "total": len(filtered)}


@router.get("/filters")  # Antes era @app.get("/api/filters")
async def get_filters():
    """Valores disponibles para filtros."""
    sentiments = sorted(set(c.get("sentiment", "") for c in chunks if c.get("sentiment")))
    categories = sorted(set(c.get("main_category", "") for c in chunks if c.get("main_category")))
    return {"sentiments": sentiments, "categories": categories}


@router.get("/eval")  # Antes era @app.get("/api/eval")
async def get_eval():
    """Reporte de evaluación de recuperación."""
    report_path = BASE_DIR / "reports" / "retrieval_eval_report.json"
    if not report_path.exists():
        return {"rows": [], "message": "Ejecuta python src/02_evaluate_retrieval.py"}

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for query_text, m in data.items():
        rows.append({
            "query": query_text[:70],
            "precision_at_k": round(m.get("precision_at_k", 0), 3),
            "mrr": round(m.get("reciprocal_rank", 0), 3),
            "cosine_sim": round(m.get("avg_cosine_sim", 0), 3),
            "latency_ms": round(m.get("latency_ms", 0), 1),
        })

    return {"rows": rows}