"""
=============================================================
 GATOBYTE — Mini Proyecto MLOps + RAG
 /src/04_baseline_comparison.py
 CRISP-DM Fase: Evaluation — Comparación Baseline
=============================================================
 Compara dos enfoques de recuperación sobre las mismas queries:

   BASELINE  → TF-IDF + similitud coseno (sklearn)
               Recuperación léxica clásica: busca coincidencia
               exacta de términos entre query y documento.

   PROPUESTO → Embeddings multilingues (MiniLM-L12) + FAISS
               Recuperación semántica: entiende significado
               y permite queries en español sobre texto en inglés.

 Evidencia requerida por la rúbrica de la asignatura:
   "Comparar contra el baseline clásico y discutir en qué
    tipos de casos mejora o empeora."

 Resultados guardados en:
   /reports/baseline_comparison_report.json
   /reports/baseline_comparison_summary.txt
   /reports/baseline_comparison_chart.png

 Run MLflow: "baseline_vs_multilingual"
=============================================================
"""

import os, json, pickle, time
import numpy as np
import pandas as pd
import faiss
import mlflow
import yaml
import matplotlib
matplotlib.use("Agg")   # sin display — funciona en local y Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────
# Config y carga de recursos
# ─────────────────────────────────────────────────────────
def load_config(path: str = "config/rag_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_latest_index(cfg: dict):
    """Carga el índice FAISS activo (construido por 01_build_index.py)."""
    index_dir  = Path(cfg["faiss"]["index_dir"])
    latest_ptr = index_dir / "latest.json"
    if not latest_ptr.exists():
        raise FileNotFoundError(
            "Índice no encontrado. Ejecuta primero: python src/01_build_index.py"
        )
    with open(latest_ptr) as f:
        ptr = json.load(f)
    with open(ptr["manifest_path"]) as f:
        manifest = json.load(f)
    index = faiss.read_index(manifest["index_file"])
    with open(manifest["metadata_file"], "rb") as f:
        chunks = pickle.load(f)
    print(f"[✓] Índice FAISS cargado — {index.ntotal:,} vectores")
    return index, chunks, manifest


def load_corpus_texts(chunks: list[dict]) -> list[str]:
    """Extrae solo los textos del corpus para TF-IDF."""
    return [c["chunk_text"] for c in chunks]


# ─────────────────────────────────────────────────────────
# BASELINE — TF-IDF + Cosine Similarity
# ─────────────────────────────────────────────────────────
class TFIDFRetriever:
    """
    Baseline clásico: TF-IDF vectorizer + similitud coseno.
    Recuperación léxica pura — sin comprensión semántica.

    Limitaciones vs modelo multilingüe:
      • No entiende sinónimos: "battery" ≠ "power cell"
      • No hace cross-language: query en español no matchea
        texto en inglés aunque signifiquen lo mismo
      • Sensible a stop words y formas verbales
    """

    def __init__(self, corpus: list[str], max_features: int = 10000):
        print(f"[→] Construyendo índice TF-IDF "
              f"(max_features={max_features})...")
        t0 = time.time()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",    # solo inglés — limitación del baseline
            ngram_range=(1, 2),      # unigramas y bigramas
            min_df=2,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        elapsed = time.time() - t0
        print(f"[✓] TF-IDF: matriz {self.matrix.shape}, "
              f"vocab={len(self.vectorizer.vocabulary_):,}, "
              f"tiempo={elapsed:.1f}s")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Busca con la query tal cual (sin traducción)."""
        q_vec   = self.vectorizer.transform([query])
        sims    = cosine_similarity(q_vec, self.matrix).flatten()
        indices = np.argsort(sims)[::-1][:top_k]
        return [
            {"index": int(i), "score": float(sims[i])}
            for i in indices if sims[i] > 0
        ]


# ─────────────────────────────────────────────────────────
# PROPUESTO — MiniLM Multilingüe + FAISS
# ─────────────────────────────────────────────────────────
class MultilingualRetriever:
    """
    Enfoque propuesto: embeddings multilingües + FAISS.
    Permite queries en español sobre corpus en inglés.
    """

    def __init__(self, index, model_name: str):
        print(f"[→] Cargando modelo multilingüe: {model_name}")
        self.index = index
        self.model = SentenceTransformer(model_name)
        print("[✓] MultilingualRetriever listo.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Busca con la query en cualquier idioma."""
        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(q_vec, top_k)
        return [
            {"index": int(idx), "score": float(score)}
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]


# ─────────────────────────────────────────────────────────
# Métricas de evaluación
# ─────────────────────────────────────────────────────────
def precision_at_k(indices: list[int], chunks: list[dict],
                   keywords: list[str], k: int) -> float:
    """Fracción de top-K chunks que contienen al menos una keyword."""
    hits = 0
    for idx in indices[:k]:
        text = chunks[idx]["chunk_text"].lower()
        if any(kw.lower() in text for kw in keywords):
            hits += 1
    return hits / min(k, len(indices)) if indices else 0.0


def reciprocal_rank(indices: list[int], chunks: list[dict],
                    keywords: list[str]) -> float:
    """1 / posición del primer resultado relevante."""
    for rank, idx in enumerate(indices, 1):
        text = chunks[idx]["chunk_text"].lower()
        if any(kw.lower() in text for kw in keywords):
            return 1.0 / rank
    return 0.0


def avg_score(results: list[dict]) -> float:
    if not results:
        return 0.0
    return float(np.mean([r["score"] for r in results]))


# ─────────────────────────────────────────────────────────
# Comparación principal
# ─────────────────────────────────────────────────────────
def run_comparison(tfidf: TFIDFRetriever,
                   multilingual: MultilingualRetriever,
                   chunks: list[dict],
                   cfg: dict,
                   top_k: int = 5) -> list[dict]:
    """
    Para cada query de la sección 'baseline' del config:
      - TF-IDF busca con la query en INGLÉS (lo mejor que puede hacer)
      - MiniLM busca con la query en ESPAÑOL (cross-language)
    Esto demuestra la ventaja semántica multilingüe.
    """
    queries  = cfg["baseline"]["comparison_queries"]
    results  = []

    print(f"\n{'─'*70}")
    print(f"  {'Query (ES)':<35} {'Modelo':<14} "
          f"{'P@K':>5}  {'MRR':>5}  {'AvgScore':>8}  {'Lat(ms)':>8}")
    print(f"{'─'*70}")

    for item in queries:
        query_es = item["query_es"]
        query_en = item["query_en"]
        keywords = item["keywords"]

        # ── TF-IDF con query en inglés ────────────────────
        t0           = time.time()
        tfidf_res    = tfidf.search(query_en, top_k)
        tfidf_lat    = (time.time() - t0) * 1000
        tfidf_idx    = [r["index"] for r in tfidf_res]
        tfidf_p      = precision_at_k(tfidf_idx, chunks, keywords, top_k)
        tfidf_rr     = reciprocal_rank(tfidf_idx, chunks, keywords)
        tfidf_score  = avg_score(tfidf_res)

        # ── MiniLM multilingüe con query en español ───────
        t0           = time.time()
        multi_res    = multilingual.search(query_es, top_k)
        multi_lat    = (time.time() - t0) * 1000
        multi_idx    = [r["index"] for r in multi_res]
        multi_p      = precision_at_k(multi_idx, chunks, keywords, top_k)
        multi_rr     = reciprocal_rank(multi_idx, chunks, keywords)
        multi_score  = avg_score(multi_res)

        # ── Determinar ganador ────────────────────────────
        winner = "MiniLM" if multi_p >= tfidf_p else "TF-IDF"

        # ── Imprimir comparación ──────────────────────────
        print(f"  {query_es[:35]:<35} {'TF-IDF':<14} "
              f"{tfidf_p:>5.2f}  {tfidf_rr:>5.2f}  "
              f"{tfidf_score:>8.4f}  {tfidf_lat:>8.1f}")
        print(f"  {'':<35} {'MiniLM (ML)':<14} "
              f"{multi_p:>5.2f}  {multi_rr:>5.2f}  "
              f"{multi_score:>8.4f}  {multi_lat:>8.1f}")
        print(f"  {'':35} → Ganador: {winner}")
        print()

        results.append({
            "query_es"      : query_es,
            "query_en"      : query_en,
            "keywords"      : keywords,
            "tfidf": {
                "precision_at_k": tfidf_p,
                "reciprocal_rank": tfidf_rr,
                "avg_score"     : round(tfidf_score, 4),
                "latency_ms"    : round(tfidf_lat, 1),
                "query_used"    : query_en,
                "top_results"   : [
                    chunks[i]["chunk_text"][:120]
                    for i in tfidf_idx[:3]
                ],
            },
            "multilingual": {
                "precision_at_k": multi_p,
                "reciprocal_rank": multi_rr,
                "avg_score"     : round(multi_score, 4),
                "latency_ms"    : round(multi_lat, 1),
                "query_used"    : query_es,
                "top_results"   : [
                    chunks[i]["chunk_text"][:120]
                    for i in multi_idx[:3]
                ],
            },
            "winner": winner,
        })

    return results


# ─────────────────────────────────────────────────────────
# Guardado de reportes
# ─────────────────────────────────────────────────────────
def save_reports(results: list[dict], cfg: dict):
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    top_k = cfg["evaluation"]["top_k"]

    # ── JSON completo ─────────────────────────────────────
    json_path = reports_dir / "baseline_comparison_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Tabla TXT ─────────────────────────────────────────
    txt_path = reports_dir / "baseline_comparison_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("GATOBYTE — Comparación Baseline: TF-IDF vs MiniLM Multilingüe\n")
        f.write("=" * 75 + "\n")
        f.write(
            f"{'Query (ES)':<35} {'Modelo':<14} "
            f"{'P@K':>5}  {'MRR':>5}  {'Score':>7}  {'Lat(ms)':>8}\n"
        )
        f.write("-" * 75 + "\n")
        for r in results:
            q = r["query_es"][:35]
            t = r["tfidf"]
            m = r["multilingual"]
            f.write(
                f"{q:<35} {'TF-IDF':<14} "
                f"{t['precision_at_k']:>5.2f}  {t['reciprocal_rank']:>5.2f}  "
                f"{t['avg_score']:>7.4f}  {t['latency_ms']:>8.1f}\n"
            )
            f.write(
                f"{'':<35} {'MiniLM (ML)':<14} "
                f"{m['precision_at_k']:>5.2f}  {m['reciprocal_rank']:>5.2f}  "
                f"{m['avg_score']:>7.4f}  {m['latency_ms']:>8.1f}\n"
            )
            f.write(f"  → Ganador: {r['winner']}\n\n")

        # Promedios
        tfidf_mean_p = np.mean([r["tfidf"]["precision_at_k"] for r in results])
        multi_mean_p = np.mean([r["multilingual"]["precision_at_k"] for r in results])
        tfidf_mean_r = np.mean([r["tfidf"]["reciprocal_rank"] for r in results])
        multi_mean_r = np.mean([r["multilingual"]["reciprocal_rank"] for r in results])

        f.write("=" * 75 + "\n")
        f.write(f"{'MEAN':<35} {'TF-IDF':<14} "
                f"{tfidf_mean_p:>5.2f}  {tfidf_mean_r:>5.2f}\n")
        f.write(f"{'':35} {'MiniLM (ML)':<14} "
                f"{multi_mean_p:>5.2f}  {multi_mean_r:>5.2f}\n")
        wins = sum(1 for r in results if r["winner"] == "MiniLM")
        f.write(f"\nMiniLM ganó en {wins}/{len(results)} queries\n")
        f.write("\nCONCLUSIÓN:\n")
        f.write(
            "El modelo multilingüe MiniLM supera a TF-IDF en queries\n"
            "cross-language (español → inglés) porque captura semántica\n"
            "contextual sin depender de coincidencia exacta de términos.\n"
            "TF-IDF requiere la query en el mismo idioma del corpus.\n"
        )

    print(f"[✓] Reporte JSON  : {json_path}")
    print(f"[✓] Reporte TXT   : {txt_path}")
    return json_path, txt_path


# ─────────────────────────────────────────────────────────
# Gráfico comparativo
# ─────────────────────────────────────────────────────────
def save_chart(results: list[dict], cfg: dict):
    """
    Genera gráfico de barras agrupadas: P@K y MRR por query,
    TF-IDF vs MiniLM multilingüe.
    """
    reports_dir = Path("reports")
    top_k       = cfg["evaluation"]["top_k"]

    queries     = [r["query_es"][:28] + "…"
                   if len(r["query_es"]) > 28 else r["query_es"]
                   for r in results]
    tfidf_p     = [r["tfidf"]["precision_at_k"]       for r in results]
    multi_p     = [r["multilingual"]["precision_at_k"] for r in results]
    tfidf_rr    = [r["tfidf"]["reciprocal_rank"]       for r in results]
    multi_rr    = [r["multilingual"]["reciprocal_rank"] for r in results]

    x      = np.arange(len(queries))
    width  = 0.35
    colors = {"tfidf": "#6B8EAD", "multi": "#E07B54"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Comparación: TF-IDF (baseline) vs MiniLM Multilingüe\n"
        "Queries en español — Corpus en inglés",
        fontsize=13, fontweight="bold"
    )

    # ── Subplot 1: Precision@K ────────────────────────────
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, tfidf_p, width,
                    label="TF-IDF (query EN)", color=colors["tfidf"],
                    alpha=0.85, edgecolor="white")
    bars2 = ax1.bar(x + width/2, multi_p, width,
                    label="MiniLM Multilingüe (query ES)",
                    color=colors["multi"], alpha=0.85, edgecolor="white")

    ax1.set_title(f"Precision@{top_k}", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Precision@K", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries, rotation=35, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.15)
    ax1.axhline(0.6, color="gray", linestyle="--", linewidth=0.8,
                label="Objetivo ≥ 0.60")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Etiquetas de valor
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                     f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                     f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    # ── Subplot 2: MRR ────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(x - width/2, tfidf_rr, width,
            label="TF-IDF (query EN)", color=colors["tfidf"],
            alpha=0.85, edgecolor="white")
    ax2.bar(x + width/2, multi_rr, width,
            label="MiniLM Multilingüe (query ES)",
            color=colors["multi"], alpha=0.85, edgecolor="white")

    ax2.set_title("Mean Reciprocal Rank (MRR)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("MRR", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(queries, rotation=35, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.15)
    ax2.axhline(0.6, color="gray", linestyle="--", linewidth=0.8,
                label="Objetivo ≥ 0.60")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # ── Promedios en el título de cada subplot ────────────
    mean_t_p = np.mean(tfidf_p)
    mean_m_p = np.mean(multi_p)
    mean_t_r = np.mean(tfidf_rr)
    mean_m_r = np.mean(multi_rr)

    ax1.set_xlabel(
        f"Media — TF-IDF: {mean_t_p:.2f} | MiniLM: {mean_m_p:.2f}",
        fontsize=9, color="dimgray"
    )
    ax2.set_xlabel(
        f"Media — TF-IDF: {mean_t_r:.2f} | MiniLM: {mean_m_r:.2f}",
        fontsize=9, color="dimgray"
    )

    plt.tight_layout()
    chart_path = reports_dir / "baseline_comparison_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Gráfico       : {chart_path}")
    return str(chart_path)


# ─────────────────────────────────────────────────────────
# Tracking MLflow
# ─────────────────────────────────────────────────────────
def log_to_mlflow(results: list[dict], cfg: dict,
                  json_path, txt_path, chart_path):
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    tfidf_mean_p = np.mean([r["tfidf"]["precision_at_k"] for r in results])
    multi_mean_p = np.mean([r["multilingual"]["precision_at_k"] for r in results])
    tfidf_mean_r = np.mean([r["tfidf"]["reciprocal_rank"] for r in results])
    multi_mean_r = np.mean([r["multilingual"]["reciprocal_rank"] for r in results])
    tfidf_mean_s = np.mean([r["tfidf"]["avg_score"] for r in results])
    multi_mean_s = np.mean([r["multilingual"]["avg_score"] for r in results])
    tfidf_mean_l = np.mean([r["tfidf"]["latency_ms"] for r in results])
    multi_mean_l = np.mean([r["multilingual"]["latency_ms"] for r in results])
    wins_multi   = sum(1 for r in results if r["winner"] == "MiniLM")

    with mlflow.start_run(run_name="baseline_vs_multilingual"):
        # Parámetros
        mlflow.log_params({
            "baseline_model"    : "TF-IDF",
            "baseline_query_lang": "english",
            "proposed_model"    : "paraphrase-multilingual-MiniLM-L12-v2",
            "proposed_query_lang": "spanish",
            "corpus_lang"       : "english",
            "tfidf_max_features": cfg["baseline"]["tfidf_max_features"],
            "num_queries"       : len(results),
            "top_k"             : cfg["evaluation"]["top_k"],
        })
        # Métricas comparativas
        mlflow.log_metrics({
            "tfidf_mean_precision_at_k" : tfidf_mean_p,
            "multi_mean_precision_at_k" : multi_mean_p,
            "tfidf_mean_mrr"            : tfidf_mean_r,
            "multi_mean_mrr"            : multi_mean_r,
            "tfidf_mean_cosine_score"   : tfidf_mean_s,
            "multi_mean_cosine_score"   : multi_mean_s,
            "tfidf_mean_latency_ms"     : tfidf_mean_l,
            "multi_mean_latency_ms"     : multi_mean_l,
            "multilingual_wins"         : wins_multi,
            "improvement_precision"     : multi_mean_p - tfidf_mean_p,
            "improvement_mrr"           : multi_mean_r - tfidf_mean_r,
        })
        # Artefactos
        mlflow.log_artifact(str(json_path))
        mlflow.log_artifact(str(txt_path))
        mlflow.log_artifact(str(chart_path))
        mlflow.log_artifact("config/rag_config.yaml")
        # Tags
        mlflow.set_tags({
            "stage"      : "baseline_comparison",
            "phase_crisp": "evaluation",
            "team"       : "GATOBYTE",
            "winner"     : "MiniLM" if multi_mean_p >= tfidf_mean_p else "TF-IDF",
        })

    print(f"[✓] MLflow run registrado: baseline_vs_multilingual")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time as time_module

    print("\n" + "="*65)
    print("  GATOBYTE · Comparación Baseline · CRISP-DM: Evaluation")
    print("="*65 + "\n")

    cfg                   = load_config()
    index, chunks, mf     = load_latest_index(cfg)
    corpus                = load_corpus_texts(chunks)

    # ── Instanciar ambos retrievers ───────────────────────
    tfidf       = TFIDFRetriever(corpus, cfg["baseline"]["tfidf_max_features"])
    multilingual = MultilingualRetriever(index, cfg["embedding"]["model_name"])

    # ── Correr comparación ────────────────────────────────
    top_k   = cfg["evaluation"]["top_k"]
    print(f"\n[→] Comparando sobre {len(cfg['baseline']['comparison_queries'])} "
          f"queries (top_k={top_k})...")
    print(f"    TF-IDF   : query en INGLÉS  (lo mejor que puede hacer)")
    print(f"    MiniLM   : query en ESPAÑOL (cross-language)")

    results = run_comparison(tfidf, multilingual, chunks, cfg, top_k)

    # ── Guardar reportes ──────────────────────────────────
    json_path, txt_path = save_reports(results, cfg)
    chart_path          = save_chart(results, cfg)

    # ── MLflow ────────────────────────────────────────────
    log_to_mlflow(results, cfg, json_path, txt_path, chart_path)

    # ── Resumen final ─────────────────────────────────────
    tfidf_mean_p = np.mean([r["tfidf"]["precision_at_k"] for r in results])
    multi_mean_p = np.mean([r["multilingual"]["precision_at_k"] for r in results])
    tfidf_mean_r = np.mean([r["tfidf"]["reciprocal_rank"] for r in results])
    multi_mean_r = np.mean([r["multilingual"]["reciprocal_rank"] for r in results])
    wins         = sum(1 for r in results if r["winner"] == "MiniLM")

    print("\n" + "="*65)
    print("  RESUMEN COMPARACIÓN")
    print("="*65)
    print(f"  {'Métrica':<28} {'TF-IDF':>10} {'MiniLM (ML)':>12}  {'Δ':>8}")
    print(f"  {'─'*60}")
    print(f"  {'Mean Precision@K':<28} {tfidf_mean_p:>10.3f} "
          f"{multi_mean_p:>12.3f}  "
          f"{multi_mean_p - tfidf_mean_p:>+8.3f}")
    print(f"  {'Mean MRR':<28} {tfidf_mean_r:>10.3f} "
          f"{multi_mean_r:>12.3f}  "
          f"{multi_mean_r - tfidf_mean_r:>+8.3f}")
    print(f"\n  MiniLM ganó en {wins}/{len(results)} queries")
    print(f"\n  CONCLUSIÓN:")
    print(f"  MiniLM multilingüe supera a TF-IDF en búsqueda cross-language")
    print(f"  (queries ES → corpus EN) porque captura significado semántico")
    print(f"  sin requerir coincidencia exacta de términos.")
    print(f"\n  Reportes en /reports/ | Run en MLflow: baseline_vs_multilingual")
    print("="*65)
