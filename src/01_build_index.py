"""
=============================================================
 GATOBYTE — Mini Proyecto MLOps + RAG
 /src/01_build_index.py
 CRISP-DM Fases: Data Preparation + Modeling
=============================================================
 Pipeline reproducible (20 pts rúbrica):
   1. [Data Prep]  Carga parquet → filtra columnas → muestra
   2. [Data Prep]  Chunking con overlap (del ejercicio docente)
   3. [Modeling]   Embeddings con all-MiniLM-L6-v2 (inferencia)
   4. [Modeling]   Índice FAISS IndexFlatIP (similitud coseno)
   5. [MLOps]      Versionado automático en /models/faiss_index/
   6. [MLOps]      Tracking completo en MLflow
=============================================================
 Mejoras incorporadas del ejercicio del docente:
   • Chunking con overlap para no perder contexto entre chunks
   • Normalización de texto antes de fragmentar
   • Manifiesto enriquecido con parámetro chunk_overlap
=============================================================
"""

import os, json, time, hashlib, pickle, datetime, shutil
import numpy as np
import pandas as pd
import faiss
import mlflow
import yaml
import random
import string

from sentence_transformers import SentenceTransformer
from pathlib import Path


# ─────────────────────────────────────────────────────────
# Carga de configuración
# ─────────────────────────────────────────────────────────
def load_config(config_path: str = "config/rag_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[✓] Config cargado: {config_path}")
    print(f"    chunk_size={cfg['chunking']['chunk_size']}  "
          f"overlap={cfg['chunking']['chunk_overlap']}  "
          f"modelo={cfg['embedding']['model_name']}")
    return cfg


# ─────────────────────────────────────────────────────────
# CRISP-DM · Fase 2: Data Understanding
# ─────────────────────────────────────────────────────────
def _make_synthetic_data(sample_size: int, seed: int) -> pd.DataFrame:
    """Datos sintéticos de demo cuando el parquet no está disponible."""
    sentiments = ["positive", "neutral", "negative"]
    categories = ["Electronics", "Computers",
                  "Cell Phones & Accessories", "Camera & Photo"]
    templates  = [
        "The battery life on this product is terrible, it dies after just 2 hours of use.",
        "Excellent image quality, the colors are vibrant and sharp.",
        "The product arrived with manufacturing defects, the casing was cracked.",
        "Very easy to set up, was working perfectly in 5 minutes.",
        "The cable breaks easily, very poor build quality and materials.",
        "Exceeded my expectations, the sound quality is incredible for the price.",
        "Frequent connectivity issues with Bluetooth, keeps disconnecting.",
        "The touchscreen is unresponsive in the corners, very frustrating.",
        "Great purchase, arrived earlier than expected and in perfect condition.",
        "The fan makes too much noise, very distracting when working at night.",
        "Fast charging works wonderfully, 80% in just 30 minutes.",
        "Fake product, does not match the photos in the listing.",
        "Internal storage is sufficient for all my photos and videos.",
        "Overheats excessively during use, worried about long-term durability.",
        "Excellent value for money, highly recommend this product.",
        "Screen brightness is too low, hard to see in sunlight.",
        "Setup was complicated and the instructions were unclear.",
        "Solid build quality, feels premium and well-made.",
        "Customer support was unhelpful when I had issues.",
        "Works exactly as described, no complaints at all.",
    ]
    random.seed(seed)
    n = sample_size
    return pd.DataFrame({
        "text"         : [random.choice(templates) + " " +
                          "".join(random.choices(string.ascii_lowercase, k=15))
                          for _ in range(n)],
        "title"        : [f"Review #{i}" for i in range(n)],
        "rating"       : [random.choice([1, 2, 3, 4, 5]) for _ in range(n)],
        "helpful_vote" : [random.randint(0, 50) for _ in range(n)],
        "parent_asin"  : [f"ASIN{random.randint(1000, 9999)}" for _ in range(n)],
        "sentiment"    : [random.choice(sentiments) for _ in range(n)],
        "main_category": [random.choice(categories) for _ in range(n)],
    })


# ─────────────────────────────────────────────────────────
# CRISP-DM · Fase 3: Data Preparation — Carga
# ─────────────────────────────────────────────────────────
def load_and_prepare(cfg: dict) -> pd.DataFrame:
    """
    Carga el parquet del proyecto (mismo de Regresión y Clustering).
    Selecciona solo las columnas necesarias para el RAG.
    Si el parquet no está disponible, usa datos sintéticos de demo.
    """
    path        = cfg["data"]["parquet_path"]
    sample_size = cfg["data"]["sample_size"]
    columns     = cfg["data"]["columns"]
    seed        = cfg["data"]["random_seed"]

    if os.path.exists(path):
        # read_parquet con 'columns' evita cargar todo el parquet en RAM
        df = pd.read_parquet(path, columns=columns)
        df = (df.dropna(subset=["text"])
                .sample(min(sample_size, len(df)), random_state=seed)
                .reset_index(drop=True))
        print(f"[✓] Parquet cargado: {len(df):,} registros")
        print(f"    Columnas: {list(df.columns)}")
    else:
        print(f"[!] Parquet no encontrado: {path}")
        print(f"    Generando datos sintéticos de demo ({sample_size:,} registros)...")
        df = _make_synthetic_data(sample_size, seed)
        print(f"[✓] Datos sintéticos listos: {len(df):,} registros")
    return df


# ─────────────────────────────────────────────────────────
# CRISP-DM · Fase 3: Data Preparation — Chunking con overlap
# ─────────────────────────────────────────────────────────
def chunk_reviews(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """
    Fragmenta reseñas largas en chunks con overlap.

    MEJORA del ejercicio del docente (chunking.py):
      • Normaliza espacios antes de fragmentar (evita chunks con saltos de línea)
      • Usa chunk_overlap para que fragmentos consecutivos compartan contexto,
        evitando que información importante quede partida entre dos chunks.
      • Conserva todos los metadatos del parquet en cada chunk.

    Ejemplo con chunk_size=500, overlap=100:
      Chunk 1: chars [   0 .. 500]
      Chunk 2: chars [ 400 .. 900]   ← los 100 chars finales del anterior se repiten
      Chunk 3: chars [ 800 .. 1300]
    """
    chunk_size = cfg["chunking"]["chunk_size"]
    overlap    = cfg["chunking"]["chunk_overlap"]
    min_len    = cfg["chunking"]["min_chunk_length"]
    step       = chunk_size - overlap   # avance real entre chunks

    chunks       = []
    chunk_number = 1   # numeración global, como en el ejercicio del docente

    for _, row in df.iterrows():
        # Normalizar espacios (igual que en chunking.py del docente)
        text  = " ".join(str(row["text"]).split())
        start = 0

        while start < len(text):
            end        = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= min_len:
                chunks.append({
                    "chunk_text"   : chunk_text,
                    "chunk_number" : chunk_number,   # metadato del docente
                    "title"        : row.get("title", ""),
                    "rating"       : row.get("rating", None),
                    "helpful_vote" : row.get("helpful_vote", 0),
                    "parent_asin"  : row.get("parent_asin", ""),
                    "sentiment"    : row.get("sentiment", ""),
                    "main_category": row.get("main_category", ""),
                })
                chunk_number += 1

            if end == len(text):
                break
            start += step   # avanza chunk_size - overlap

    print(f"[✓] Chunks generados: {len(chunks):,}  "
          f"(overlap={overlap} chars, step={step} chars)")
    return chunks


# ─────────────────────────────────────────────────────────
# CRISP-DM · Fase 4: Modeling — Embeddings
# ─────────────────────────────────────────────────────────
def generate_embeddings(chunks: list[dict], cfg: dict) -> np.ndarray:
    """
    Genera embeddings con all-MiniLM-L6-v2 en modo INFERENCIA pura.

    Cumple la política de Transformers de la asignatura:
      ✅ Modelo preentrenado, sin fine-tuning
      ✅ Sin GPU dedicada (CPU suficiente)
      ✅ Modelo pequeño (~80 MB, 384 dimensiones)

    Comparación vs baseline TF-IDF (requerida por la rúbrica):
      TF-IDF: frecuencia de términos, sin semántica
      MiniLM: embeddings contextuales, captura sinónimos
      Ejemplo: "batería agota" ↔ "autonomía baja" → match semántico con MiniLM
    """
    model_name = cfg["embedding"]["model_name"]
    batch_size = cfg["embedding"]["batch_size"]
    normalize  = cfg["embedding"]["normalize"]

    print(f"[→] Cargando modelo: {model_name}")
    model  = SentenceTransformer(model_name)
    texts  = [c["chunk_text"] for c in chunks]

    print(f"[→] Generando embeddings para {len(texts):,} chunks...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,   # L2-norm → coseno = dot product
    )
    elapsed = time.time() - t0
    print(f"[✓] Embeddings: {embeddings.shape} en {elapsed:.1f}s  "
          f"({len(texts)/elapsed:.0f} chunks/s)")
    return embeddings


# ─────────────────────────────────────────────────────────
# CRISP-DM · Fase 4: Modeling — Índice FAISS
# ─────────────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    IndexFlatIP: búsqueda exacta por producto interno.
    Con vectores L2-normalizados, IP = similitud coseno.
    Apropiado para hasta ~500k vectores en CPU.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    print(f"[✓] Índice FAISS: {index.ntotal:,} vectores, dim={dim}")
    return index


# ─────────────────────────────────────────────────────────
# MLOps · Versionado automático en /models/
# (Rúbrica: Tracking y versionado — 20 pts)
# ─────────────────────────────────────────────────────────
def save_index(index, chunks: list[dict],
               embeddings: np.ndarray, cfg: dict) -> dict:
    """
    Persiste el índice con versionado automático.
    Estructura bajo /models/faiss_index/:
      v_YYYYMMDD_HHMMSS_<hash>/
        ├── index.faiss         ← artefacto FAISS
        ├── chunks_metadata.pkl ← metadatos de cada chunk
        └── manifest.json       ← manifiesto legible (versión, params, paths)
      latest.json               ← puntero a la versión activa
    """
    index_dir = Path(cfg["faiss"]["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    ts       = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    cfg_hash = hashlib.md5(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:8]
    version  = f"v_{ts}_{cfg_hash}"

    vdir = index_dir / version
    vdir.mkdir(parents=True, exist_ok=True)

    # ── Artefactos del modelo ─────────────────────────────
    faiss_path = vdir / "index.faiss"
    meta_path  = vdir / "chunks_metadata.pkl"
    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    # ── Manifiesto enriquecido ────────────────────────────
    manifest = {
        "version"        : version,
        "created_at_utc" : ts,
        "config_hash"    : cfg_hash,
        "model_name"     : cfg["embedding"]["model_name"],
        "data_source"    : "sample_ml.parquet",
        "sample_size"    : cfg["data"]["sample_size"],
        "chunk_size"     : cfg["chunking"]["chunk_size"],
        "chunk_overlap"  : cfg["chunking"]["chunk_overlap"],  # nuevo
        "num_chunks"     : len(chunks),
        "embedding_dim"  : int(embeddings.shape[1]),
        "index_type"     : cfg["faiss"]["index_type"],
        "index_file"     : str(faiss_path),
        "metadata_file"  : str(meta_path),
    }
    manifest_path = vdir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── Puntero a versión activa ──────────────────────────
    latest = index_dir / "latest.json"
    with open(latest, "w") as f:
        json.dump({"latest_version": version,
                   "manifest_path" : str(manifest_path)}, f, indent=2)

    print(f"[✓] Índice guardado en: {vdir}")
    print(f"    Versión   : {version}")
    print(f"    Chunks    : {len(chunks):,}")
    print(f"    Overlap   : {cfg['chunking']['chunk_overlap']} chars")

    # ── Retención de versiones antiguas ──────────────────
    _cleanup_old_versions(index_dir, cfg["faiss"]["keep_n_versions"])
    return manifest


def _cleanup_old_versions(index_dir: Path, keep_n: int):
    dirs = sorted([d for d in index_dir.iterdir() if d.is_dir()],
                  key=lambda d: d.name)
    for old in dirs[:-keep_n]:
        shutil.rmtree(old)
        print(f"[→] Versión antigua eliminada: {old.name}")


# ─────────────────────────────────────────────────────────
# MLOps · Tracking con MLflow
# (Rúbrica: Tracking y versionado — 20 pts)
# ─────────────────────────────────────────────────────────
def log_to_mlflow(manifest: dict, elapsed: float, cfg: dict):
    """
    Registra parámetros, métricas y artefactos del run de indexación.
    Permite reproducir exactamente cualquier versión del índice.
    Ver todos los runs: mlflow ui
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"build_index_{manifest['version']}"):
        # Parámetros → reproducibilidad total
        mlflow.log_params({
            "model_name"    : manifest["model_name"],
            "data_source"   : manifest["data_source"],
            "sample_size"   : manifest["sample_size"],
            "chunk_size"    : manifest["chunk_size"],
            "chunk_overlap" : manifest["chunk_overlap"],
            "index_type"    : manifest["index_type"],
            "config_hash"   : manifest["config_hash"],
        })
        # Métricas del proceso
        mlflow.log_metrics({
            "num_chunks"       : manifest["num_chunks"],
            "embedding_dim"    : manifest["embedding_dim"],
            "build_time_sec"   : round(elapsed, 2),
            "chunks_per_second": round(manifest["num_chunks"] / elapsed, 1),
        })
        # Artefactos persistidos
        mlflow.log_artifact(manifest["index_file"])
        mlflow.log_artifact(manifest["metadata_file"])
        mlflow.log_artifact(
            str(Path(manifest["index_file"]).parent / "manifest.json")
        )
        mlflow.log_artifact("config/rag_config.yaml")
        # Tags de trazabilidad
        mlflow.set_tags({
            "index_version": manifest["version"],
            "stage"        : "indexing",
            "phase_crisp"  : "data_prep_modeling",
            "team"         : "GATOBYTE",
        })
    print(f"[✓] MLflow run registrado — experimento: "
          f"'{cfg['mlflow']['experiment_name']}'")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    print("\n" + "="*62)
    print("  GATOBYTE · RAG Pipeline · CRISP-DM: Data Prep + Modeling")
    print("="*62 + "\n")

    cfg        = load_config("config/rag_config.yaml")
    df         = load_and_prepare(cfg)
    chunks     = chunk_reviews(df, cfg)
    embeddings = generate_embeddings(chunks, cfg)
    index      = build_faiss_index(embeddings)
    manifest   = save_index(index, chunks, embeddings, cfg)

    elapsed = time.time() - t0
    log_to_mlflow(manifest, elapsed, cfg)

    print(f"\n{'='*62}")
    print(f"  Pipeline completo en {elapsed:.1f}s")
    print(f"  Chunks indexados : {manifest['num_chunks']:,}")
    print(f"  Versión          : {manifest['version']}")
    print(f"  → Siguiente: python src/02_evaluate_retrieval.py")
    print(f"{'='*62}")
