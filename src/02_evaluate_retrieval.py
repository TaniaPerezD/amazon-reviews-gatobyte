"""
=============================================================
 GATOBYTE — Mini Proyecto MLOps + RAG
 /src/02_evaluate_retrieval.py
 CRISP-DM Fase: Evaluation
=============================================================
 Rúbrica cubierta:
   • Recuperación semántica (25 pts) — SemanticRetriever con
     prompt estructurado (adaptado del ejercicio del docente)
   • Tracking y versionado  (20 pts) — MLflow con métricas
     por query y métricas globales
=============================================================
 Mejoras del ejercicio del docente incorporadas:
   • Prompt estructurado con contexto numerado por fuente
   • Metadatos de fuente en cada resultado (chunk_number,
     sentiment, main_category, rating)
   • Reporte de fuentes deduplicadas (como _sources_from_chunks)
   • MRR (Mean Reciprocal Rank) añadido como métrica extra
=============================================================
"""

import json, pickle, time
import numpy as np
import faiss
import mlflow
import yaml

from pathlib import Path
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────
# Carga de recursos
# ─────────────────────────────────────────────────────────
def load_config(path: str = "config/rag_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_latest_index(cfg: dict):
    """Carga la versión activa desde /models/faiss_index/latest.json"""
    index_dir  = Path(cfg["faiss"]["index_dir"])
    latest_ptr = index_dir / "latest.json"

    if not latest_ptr.exists():
        raise FileNotFoundError(
            "No se encontró latest.json. "
            "Ejecuta primero: python src/01_build_index.py"
        )
    with open(latest_ptr) as f:
        ptr = json.load(f)
    with open(ptr["manifest_path"]) as f:
        manifest = json.load(f)

    index = faiss.read_index(manifest["index_file"])
    with open(manifest["metadata_file"], "rb") as f:
        chunks = pickle.load(f)

    print(f"[✓] Índice cargado — versión: {manifest['version']}")
    print(f"    Vectores : {index.ntotal:,} | Dim: {index.d}")
    print(f"    Overlap  : {manifest.get('chunk_overlap', 'N/A')} chars")
    return index, chunks, manifest


# ─────────────────────────────────────────────────────────
# CRISP-DM · Fase 5: Evaluation — Motor de recuperación
# ─────────────────────────────────────────────────────────
class SemanticRetriever:
    """
    Motor de recuperación semántica.
    Incorpora el patrón del ejercicio del docente:
      • search() devuelve chunks con metadatos completos
      • sources() deduplica fuentes (como _sources_from_chunks en rag.py)
      • build_context() formatea el contexto numerado para el prompt
    """

    def __init__(self, index, chunks: list, cfg: dict):
        self.index  = index
        self.chunks = chunks
        model_name  = cfg["embedding"]["model_name"]
        print(f"[→] Cargando modelo de embeddings: {model_name}")
        self.model  = SentenceTransformer(model_name)
        print("[✓] SemanticRetriever listo.")

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.0) -> list[dict]:
        """
        Busca los top_k fragmentos más similares a la query.
        Retorna lista con score coseno y todos los metadatos del chunk.
        """
        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Pedimos top_k*3 para poder filtrar por min_score
        scores, indices = self.index.search(q_vec, top_k * 3)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or float(score) < min_score:
                continue
            c = self.chunks[idx]
            results.append({
                "score"        : float(score),
                "chunk_text"   : c["chunk_text"],
                "chunk_number" : c.get("chunk_number", 0),
                "rating"       : c.get("rating"),
                "helpful_vote" : c.get("helpful_vote", 0),
                "parent_asin"  : c.get("parent_asin", ""),
                "title"        : c.get("title", ""),
                "sentiment"    : c.get("sentiment", ""),
                "main_category": c.get("main_category", ""),
            })
            if len(results) == top_k:
                break
        return results

    def sources_from_results(self, results: list[dict]) -> list[dict]:
        """
        Deduplica fuentes de los chunks recuperados.
        Adaptado de _sources_from_chunks() del ejercicio del docente.
        """
        seen    = set()
        sources = []
        for r in results:
            key = (r["parent_asin"], r["chunk_number"])
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "parent_asin"  : r["parent_asin"],
                "title"        : r["title"],
                "chunk_number" : r["chunk_number"],
                "sentiment"    : r["sentiment"],
                "main_category": r["main_category"],
                "rating"       : r["rating"],
                "score"        : round(r["score"], 4),
            })
        return sources

    def build_context(self, results: list[dict]) -> str:
        """
        Formatea el contexto numerado para el prompt RAG.
        Adaptado de _build_prompt() del ejercicio del docente.
        """
        parts = []
        for idx, r in enumerate(results, 1):
            parts.append(
                f"[Fuente {idx}] "
                f"ASIN={r['parent_asin']} | "
                f"chunk={r['chunk_number']} | "
                f"sentiment={r['sentiment']} | "
                f"rating={r['rating']}\n"
                f"{r['chunk_text']}"
            )
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────
# Métricas de evaluación
# ─────────────────────────────────────────────────────────
def precision_at_k(results: list[dict], keywords: list[str], k: int) -> float:
    """
    Precision@K: fracción de los top-K resultados que contienen
    al menos una de las keywords esperadas.
    Métrica verificable sin etiquetas humanas costosas.
    """
    hits = sum(
        1 for r in results[:k]
        if any(kw.lower() in r["chunk_text"].lower() for kw in keywords)
    )
    return hits / min(k, len(results)) if results else 0.0


def reciprocal_rank(results: list[dict], keywords: list[str]) -> float:
    """
    MRR (Mean Reciprocal Rank): posición del primer resultado relevante.
    MRR = 1/posición_primer_hit  (1.0 = primer resultado es relevante)
    Métrica complementaria a Precision@K.
    """
    for rank, r in enumerate(results, 1):
        if any(kw.lower() in r["chunk_text"].lower() for kw in keywords):
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────
# Evaluación formal con MLflow
# ─────────────────────────────────────────────────────────
def run_evaluation(retriever: SemanticRetriever, cfg: dict) -> dict:
    """
    Ejecuta las queries de evaluación del config.
    Calcula Precision@K, MRR, similitud coseno y latencia.
    Registra todo en MLflow para evidencia de tracking.
    """
    top_k   = cfg["evaluation"]["top_k"]
    queries = cfg["evaluation"]["queries"]

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    prec_scores, rr_scores, latencies, report = [], [], [], {}

    with mlflow.start_run(run_name="eval_retrieval_quality"):
        for item in queries:
            query    = item["query"]
            keywords = item["keywords"]

            t0      = time.time()
            results = retriever.search(query, top_k=top_k)
            lat     = time.time() - t0

            p_at_k    = precision_at_k(results, keywords, top_k)
            rr        = reciprocal_rank(results, keywords)
            avg_score = float(np.mean([r["score"] for r in results])) if results else 0.0
            sources   = retriever.sources_from_results(results)

            prec_scores.append(p_at_k)
            rr_scores.append(rr)
            latencies.append(lat)

            report[query] = {
                "precision_at_k": p_at_k,
                "reciprocal_rank": rr,
                "avg_cosine_sim" : round(avg_score, 4),
                "latency_ms"     : round(lat * 1000, 1),
                "num_sources"    : len(sources),
                "results"        : results,
                "sources"        : sources,
            }

            # Log por query en MLflow
            safe = query[:35].replace(" ", "_")
            mlflow.log_metric(f"precision_at_{top_k}_{safe}", p_at_k)
            mlflow.log_metric(f"reciprocal_rank_{safe}", rr)
            mlflow.log_metric(f"avg_cosine_{safe}", avg_score)
            mlflow.log_metric(f"latency_ms_{safe}", lat * 1000)

        # Métricas globales
        mean_prec = float(np.mean(prec_scores))
        mean_mrr  = float(np.mean(rr_scores))
        mean_lat  = float(np.mean(latencies)) * 1000

        mlflow.log_metrics({
            "mean_precision_at_k": mean_prec,
            "mean_mrr"           : mean_mrr,
            "mean_latency_ms"    : mean_lat,
            "num_eval_queries"   : len(queries),
            "top_k"              : top_k,
        })
        mlflow.set_tags({
            "stage"      : "evaluation",
            "phase_crisp": "evaluation",
            "team"       : "GATOBYTE",
        })

    print(f"\n[✓] Evaluación completada")
    print(f"    Mean Precision@{top_k}: {mean_prec:.3f}")
    print(f"    Mean MRR             : {mean_mrr:.3f}")
    print(f"    Mean Latencia        : {mean_lat:.1f} ms")
    return report


# ─────────────────────────────────────────────────────────
# Guardado de reportes en /reports/
# ─────────────────────────────────────────────────────────
def save_report(report: dict, cfg: dict):
    """
    Guarda JSON + tabla TXT en /reports/ para el informe final.
    También registra los reportes como artefactos MLflow.
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # ── JSON limpio (sin listas de resultados completas) ──
    json_path = reports_dir / "retrieval_eval_report.json"
    clean = {
        q: {k: v for k, v in m.items() if k not in ("results", "sources")}
        for q, m in report.items()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    # ── Tabla resumen TXT ─────────────────────────────────
    top_k    = cfg["evaluation"]["top_k"]
    txt_path = reports_dir / "retrieval_eval_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("GATOBYTE — Evaluación de Recuperación Semántica (RAG)\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'Query':<44} {'P@K':>5}  {'MRR':>5}  "
                f"{'CosSim':>7}  {'Lat(ms)':>8}\n")
        f.write("-" * 72 + "\n")
        for query, m in report.items():
            f.write(
                f"{query[:44]:<44} "
                f"{m['precision_at_k']:>5.2f}  "
                f"{m['reciprocal_rank']:>5.2f}  "
                f"{m['avg_cosine_sim']:>7.4f}  "
                f"{m['latency_ms']:>8.1f}\n"
            )
        vals = list(report.values())
        f.write("-" * 72 + "\n")
        f.write(
            f"{'MEAN':<44} "
            f"{np.mean([v['precision_at_k'] for v in vals]):>5.2f}  "
            f"{np.mean([v['reciprocal_rank'] for v in vals]):>5.2f}  "
            f"{'—':>7}  "
            f"{np.mean([v['latency_ms'] for v in vals]):>8.1f}\n"
        )

    # ── Fuentes por query ─────────────────────────────────
    sources_path = reports_dir / "retrieval_sources.json"
    sources_data = {q: m["sources"] for q, m in report.items()}
    with open(sources_path, "w", encoding="utf-8") as f:
        json.dump(sources_data, f, indent=2, ensure_ascii=False)

    # Registrar en MLflow como artefactos
    try:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="save_eval_reports"):
            mlflow.log_artifact(str(json_path))
            mlflow.log_artifact(str(txt_path))
            mlflow.log_artifact(str(sources_path))
    except Exception:
        pass   # no interrumpir si MLflow falla aquí

    print(f"[✓] Reportes guardados en /{reports_dir}/")
    return str(json_path), str(txt_path)


# ─────────────────────────────────────────────────────────
# Fix 1 — Ejemplos documentados con texto real recuperado
# ─────────────────────────────────────────────────────────
def save_query_examples(report: dict, cfg: dict) -> str:
    """
    Genera reports/query_examples.md con el texto real de los chunks
    recuperados para cada query. Cubre el requisito de 'al menos cinco
    ejemplos de consulta documentados' con evidencia legible.
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    top_k   = cfg["evaluation"]["top_k"]
    model   = cfg["embedding"]["model_name"]
    idx_type = cfg["faiss"]["index_type"]
    md_path = reports_dir / "query_examples.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# GATOBYTE — Ejemplos Documentados de Recuperación Semántica\n\n")
        f.write(f"**Modelo:** `{model}`  \n")
        f.write(f"**Índice:** FAISS `{idx_type}`  \n")
        f.write(f"**Top-K:** {top_k}  \n")
        f.write(f"**Queries en español — corpus en inglés (cross-language)**\n\n")
        f.write("---\n\n")

        for i, (query, m) in enumerate(report.items(), 1):
            f.write(f"## Ejemplo {i} — \"{query}\"\n\n")
            f.write(f"| Métrica | Valor |\n|---|---|\n")
            f.write(f"| Precision@{top_k} | **{m['precision_at_k']:.2f}** |\n")
            f.write(f"| MRR | **{m['reciprocal_rank']:.2f}** |\n")
            f.write(f"| Cosine Sim promedio | {m['avg_cosine_sim']:.4f} |\n")
            f.write(f"| Latencia | {m['latency_ms']:.1f} ms |\n\n")

            results = m.get("results", [])
            if results:
                f.write(f"### Fragmentos recuperados\n\n")
                for j, r in enumerate(results, 1):
                    level = ("Alta" if r["score"] >= 0.6
                             else ("Media" if r["score"] >= 0.4 else "Baja"))
                    f.write(
                        f"**[{j}]** Similitud: `{r['score']:.4f}` ({level}) | "
                        f"Rating: {r['rating']} ★ | "
                        f"Sentiment: `{r['sentiment']}` | "
                        f"Categoría: `{r['main_category']}` | "
                        f"ASIN: `{r['parent_asin']}`\n\n"
                    )
                    text = r["chunk_text"]
                    if len(text) > 420:
                        text = text[:420] + "..."
                    f.write(f"> {text}\n\n")
            else:
                f.write("_Sin resultados._\n\n")

            f.write("---\n\n")

    print(f"[✓] Ejemplos documentados → reports/query_examples.md  ({len(report)} queries)")
    return str(md_path)


# ─────────────────────────────────────────────────────────
# Fix 2 — Exportar historial de runs de MLflow a CSV
# ─────────────────────────────────────────────────────────
def export_mlflow_runs(cfg: dict) -> str | None:
    """
    Exporta todos los runs del experimento MLflow a un CSV legible.
    Permite evidenciar el tracking sin necesidad de mlflow ui.
    """
    import pandas as pd

    try:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        runs = mlflow.search_runs(
            experiment_names=[cfg["mlflow"]["experiment_name"]],
            order_by=["start_time DESC"],
        )
        if runs.empty:
            print("[!] No se encontraron runs de MLflow — ejecuta primero los scripts.")
            return None

        wanted = [
            "run_id", "status", "start_time",
            "tags.mlflow.runName", "tags.stage", "tags.update_decision",
            "metrics.mean_precision_at_k", "metrics.mean_mrr",
            "metrics.mean_latency_ms", "metrics.num_chunks",
            "metrics.build_time_sec", "metrics.index_age_days",
            "metrics.avg_cosine_control", "metrics.needs_update",
            "params.model_name", "params.chunk_size", "params.chunk_overlap",
            "params.sample_size", "params.rebuild_strategy",
        ]
        cols = [c for c in wanted if c in runs.columns]
        export = runs[cols].copy()
        export.columns = [
            c.replace("tags.", "").replace("metrics.", "").replace("params.", "")
            for c in cols
        ]

        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        csv_path = reports_dir / "mlflow_runs_summary.csv"
        export.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[✓] MLflow runs exportados: {len(runs)} runs → reports/mlflow_runs_summary.csv")
        return str(csv_path)

    except Exception as e:
        print(f"[!] Error exportando MLflow: {e}")
        return None


# ─────────────────────────────────────────────────────────
# Utilidades de visualización en consola
# ─────────────────────────────────────────────────────────
def print_results(query: str, results: list[dict], retriever: SemanticRetriever):
    print(f"\n{'─'*65}")
    print(f"  Query: «{query}»")
    print(f"{'─'*65}")
    for i, r in enumerate(results, 1):
        badge = "🟢" if r["score"] >= 0.6 else ("🟡" if r["score"] >= 0.4 else "🔴")
        print(f"\n  {badge} [{i}] Score: {r['score']:.4f} | "
              f"Rating: {r['rating']} ★ | Votos: {r['helpful_vote']}")
        print(f"      Sentiment: {r['sentiment']} | "
              f"Categoría: {r['main_category']} | "
              f"Chunk #{r['chunk_number']}")
        print(f"      ASIN: {r['parent_asin']}")
        print(f"      › {r['chunk_text'][:200]}...")

    sources = retriever.sources_from_results(results)
    print(f"\n  Fuentes únicas recuperadas: {len(sources)}")
    context = retriever.build_context(results)
    print(f"  Tamaño del contexto: {len(context)} chars")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*62)
    print("  GATOBYTE · RAG Evaluation · CRISP-DM: Evaluation")
    print("="*62 + "\n")

    cfg                  = load_config()
    index, chunks, mf    = load_latest_index(cfg)
    retriever            = SemanticRetriever(index, chunks, cfg)

    # ── Demo de búsqueda con 3 queries obligatorias ───────
    demo_queries = [
        "problemas frecuentes con el producto",
        "opiniones sobre batería",
        "defectos de fabricación",
    ]
    print("\n[DEMO DE BÚSQUEDA SEMÁNTICA]")
    for q in demo_queries:
        results = retriever.search(q, top_k=cfg["retrieval"]["default_top_k"])
        print_results(q, results, retriever)

    # ── Evaluación formal con todas las queries del config ─
    print("\n\n[→] Ejecutando evaluación formal con MLflow...")
    report = run_evaluation(retriever, cfg)
    save_report(report, cfg)
    save_query_examples(report, cfg)
    export_mlflow_runs(cfg)

    # ── Resumen final ─────────────────────────────────────
    print("\n\n[RESUMEN DE EVALUACIÓN]")
    print(f"{'─'*72}")
    top_k = cfg["evaluation"]["top_k"]
    for query, m in report.items():
        print(f"  {query[:50]:<50} "
              f"P@{top_k}={m['precision_at_k']:.2f}  "
              f"MRR={m['reciprocal_rank']:.2f}  "
              f"cos={m['avg_cosine_sim']:.3f}  "
              f"lat={m['latency_ms']:.0f}ms")

    vals = list(report.values())
    print(f"{'─'*72}")
    print(f"  {'MEAN':<50} "
          f"P@{top_k}={np.mean([v['precision_at_k'] for v in vals]):.2f}  "
          f"MRR={np.mean([v['reciprocal_rank'] for v in vals]):.2f}")
    print(f"\n[✓] Reportes en /reports/ | Tracking en MLflow")
    print(f"    → reports/query_examples.md       (texto real recuperado)")
    print(f"    → reports/mlflow_runs_summary.csv (historial de experimentos)")
    print(f"→ Siguiente: uvicorn demo.main:app --reload --port 7860")
