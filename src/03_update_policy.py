"""
=============================================================
 GATOBYTE — Mini Proyecto MLOps + RAG
 /src/03_update_policy.py
 CRISP-DM Fase: Deployment — Monitoreo continuo
 (Rúbrica: Monitoreo y reentrenamiento — 15 pts)
=============================================================
 Implementa 3 triggers de re-indexación:
   T1 — Antigüedad del índice (> max_age_days)
   T2 — Drift de calidad     (cosine sim < umbral)
   T3 — Volumen datos nuevos (> umbral %)
 Decisión KEEP / REBUILD registrada en MLflow.
 Limpieza automática de versiones antiguas.
=============================================================
"""

import json, pickle, time, datetime, shutil
import numpy as np
import faiss
import mlflow
import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime as dt
from pathlib import Path
from sentence_transformers import SentenceTransformer


def load_config(path: str = "config/rag_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────
# Carga del manifiesto de la versión activa
# ─────────────────────────────────────────────────────────
def load_manifest(cfg: dict) -> dict | None:
    index_dir  = Path(cfg["faiss"]["index_dir"])
    latest_ptr = index_dir / "latest.json"
    if not latest_ptr.exists():
        print("[!] No hay índice previo. Ejecuta primero: python src/01_build_index.py")
        return None
    with open(latest_ptr) as f:
        ptr = json.load(f)
    with open(ptr["manifest_path"]) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────
# T1 — Antigüedad del índice
# ─────────────────────────────────────────────────────────
def check_age(manifest: dict, cfg: dict) -> tuple[bool, float]:
    """Retorna (trigger_activo, edad_en_días)."""
    ts         = manifest.get("created_at_utc", "")
    created_dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")
    age_days   = (datetime.datetime.utcnow() - created_dt).total_seconds() / 86400
    limit      = cfg["update_policy"]["max_age_days"]
    return age_days > limit, round(age_days, 2)


# ─────────────────────────────────────────────────────────
# T2 — Drift de calidad (similitud coseno)
# ─────────────────────────────────────────────────────────
def check_quality_drift(manifest: dict, cfg: dict) -> tuple[bool, float]:
    """
    Evalúa queries de control sobre el índice activo.
    Si la similitud coseno promedio cae bajo el umbral → drift detectado.
    Mismas queries en inglés/español para cubrir el dataset bilingüe.
    """
    CONTROL_QUERIES = [
        "battery life too short",
        "defective product arrived broken",
        "excellent image quality",
        "easy to set up and install",
        "poor build quality materials",
        "batería dura poco",
        "producto defectuoso",
    ]
    try:
        index = faiss.read_index(manifest["index_file"])
        with open(manifest["metadata_file"], "rb") as f:
            _ = pickle.load(f)   # valida que no está corrupto

        model = SentenceTransformer(cfg["embedding"]["model_name"])
        sims  = []
        for q in CONTROL_QUERIES:
            q_vec = model.encode(
                [q], convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32")
            scores, _ = index.search(q_vec, 3)
            sims.append(float(np.mean(scores[0])))

        avg_sim = float(np.mean(sims))
        limit   = cfg["update_policy"]["min_avg_cosine_sim"]
        return avg_sim < limit, round(avg_sim, 4)

    except Exception as e:
        print(f"[!] Error en check_quality_drift: {e}")
        return False, -1.0


# ─────────────────────────────────────────────────────────
# T3 — Volumen de datos nuevos (funcional)
# ─────────────────────────────────────────────────────────
def check_new_data(manifest: dict, cfg: dict) -> tuple[bool, float]:
    """
    Lee data/new_records_count.txt para saber cuántos registros nuevos
    han llegado desde la última indexación.

    Flujo de producción:
      • Un proceso ETL actualiza new_records_count.txt al ingestar datos.
      • Este script lee el contador y decide si el volumen justifica REBUILD.
      • Tras un REBUILD exitoso el contador debe resetearse a 0.

    Fallback: si el archivo no existe, compara la fecha de modificación
    del parquet vs la fecha de creación del índice como estimación.
    """
    counter_path = Path("data/new_records_count.txt")
    current_size = manifest.get("sample_size", 1)
    threshold    = cfg["update_policy"]["new_data_threshold_pct"]
    new_records  = 0

    if counter_path.exists():
        try:
            new_records = int(counter_path.read_text().strip())
            print(f"    [T3] Contador leído: {counter_path} → {new_records:,} registros nuevos")
        except (ValueError, IOError):
            new_records = 0
    else:
        # Fallback: detectar si el parquet fue modificado después del índice
        parquet_path = Path(cfg["data"]["parquet_path"])
        if parquet_path.exists():
            try:
                ts         = manifest.get("created_at_utc", "")
                created_dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")
                pmtime     = datetime.datetime.utcfromtimestamp(
                    parquet_path.stat().st_mtime
                )
                if pmtime > created_dt:
                    new_records = int(current_size * 0.05)
                    print(f"    [T3] Parquet más nuevo que el índice — "
                          f"estimando {new_records:,} registros nuevos")
                else:
                    print(f"    [T3] Sin archivo contador y parquet sin cambios → 0")
            except Exception:
                pass
        else:
            print(f"    [T3] Parquet no encontrado → asumiendo 0 registros nuevos")

    pct_new = (new_records / current_size) * 100 if current_size > 0 else 0.0
    return pct_new > threshold, round(pct_new, 2)


# ─────────────────────────────────────────────────────────
# Limpieza de versiones antiguas
# ─────────────────────────────────────────────────────────
def cleanup_old_versions(cfg: dict):
    index_dir = Path(cfg["faiss"]["index_dir"])
    keep_n    = cfg["faiss"]["keep_n_versions"]
    dirs = sorted([d for d in index_dir.iterdir() if d.is_dir()],
                  key=lambda d: d.name)
    deleted = 0
    for old in dirs[:-keep_n]:
        shutil.rmtree(old)
        print(f"[→] Versión eliminada: {old.name}")
        deleted += 1
    kept = len(dirs) - deleted
    print(f"[✓] Retención: {kept} versión(es) conservada(s) "
          f"(política: keep_n={keep_n})")


# ─────────────────────────────────────────────────────────
# Registro de la decisión en MLflow
# ─────────────────────────────────────────────────────────
def log_policy_decision(manifest: dict, age_days: float, avg_sim: float,
                        pct_new: float, needs_update: bool,
                        reasons: list[str], cfg: dict):
    """
    Registra la decisión KEEP/REBUILD en MLflow con todos los valores
    de los triggers, para trazabilidad histórica del monitoreo.
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")
    with mlflow.start_run(run_name=f"policy_check_{ts}"):
        mlflow.log_params({
            "checked_version"       : manifest.get("version", "unknown"),
            "max_age_days_policy"   : cfg["update_policy"]["max_age_days"],
            "min_cosine_policy"     : cfg["update_policy"]["min_avg_cosine_sim"],
            "new_data_threshold_pct": cfg["update_policy"]["new_data_threshold_pct"],
            "rebuild_strategy"      : cfg["update_policy"]["rebuild_strategy"],
        })
        mlflow.log_metrics({
            "index_age_days"    : age_days,
            "avg_cosine_control": avg_sim,
            "pct_new_data"      : pct_new,
            "needs_update"      : int(needs_update),
        })
        mlflow.set_tags({
            "stage"          : "monitoring",
            "phase_crisp"    : "deployment",
            "update_decision": "REBUILD" if needs_update else "KEEP",
            "trigger_reasons": "; ".join(reasons) if reasons else "none",
            "team"           : "GATOBYTE",
        })
    print(f"[✓] Decisión registrada en MLflow: "
          f"{'⚠ REBUILD' if needs_update else '✅ KEEP'}")


# ─────────────────────────────────────────────────────────
# Mejora 2 — Historial persistente de monitoreo
# ─────────────────────────────────────────────────────────
def save_monitoring_history(manifest: dict, age_days: float, avg_sim: float,
                             pct_new: float, needs_update: bool,
                             reasons: list[str]) -> str:
    """
    Persiste cada chequeo en reports/monitoring_history.json.
    Permite analizar la tendencia de drift a lo largo del tiempo
    y alimenta la gráfica plot_drift_trend().
    """
    history_path = Path("reports/monitoring_history.json")
    history: list = []

    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)

    history.append({
        "checked_at_utc"    : datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "index_version"     : manifest.get("version", "unknown"),
        "age_days"          : round(age_days, 2),
        "avg_cosine_control": avg_sim,
        "pct_new_data"      : pct_new,
        "decision"          : "REBUILD" if needs_update else "KEEP",
        "triggers"          : reasons if reasons else [],
    })

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"[✓] Historial actualizado: {len(history)} chequeo(s) "
          f"en reports/monitoring_history.json")
    return str(history_path)


# ─────────────────────────────────────────────────────────
# Mejora 3 — Gráfica de tendencia de drift
# ─────────────────────────────────────────────────────────
def plot_drift_trend(cfg: dict) -> str | None:
    """
    Lee reports/monitoring_history.json y genera reports/drift_trend.png.
    Muestra la evolución de la similitud coseno con el umbral de rebuild
    y las decisiones KEEP / REBUILD marcadas visualmente.
    """
    history_path = Path("reports/monitoring_history.json")
    if not history_path.exists():
        print("[!] Sin historial aún — omitiendo gráfica.")
        return None

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        print("[!] Historial vacío — omitiendo gráfica.")
        return None

    dates     = [dt.strptime(h["checked_at_utc"], "%Y-%m-%d %H:%M:%S") for h in history]
    cosines   = [h["avg_cosine_control"] for h in history]
    pct_news  = [h["pct_new_data"] for h in history]
    decisions = [h["decision"] for h in history]
    threshold_cos  = cfg["update_policy"]["min_avg_cosine_sim"]
    threshold_data = cfg["update_policy"]["new_data_threshold_pct"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax2 = ax.twinx()

    # Eje izquierdo — similitud coseno
    ax.plot(dates, cosines, "b-o", markersize=7, linewidth=2,
            label="Cosine Sim (eje izq.)")
    ax.axhline(y=threshold_cos, color="blue", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"Umbral coseno ({threshold_cos})")
    ax.axhspan(0, threshold_cos, alpha=0.05, color="blue")
    ax.set_ylabel("Similitud coseno promedio", color="blue")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelcolor="blue")

    # Eje derecho — % datos nuevos
    ax2.bar(dates, pct_news, width=0.003, alpha=0.35, color="orange",
            label=f"% datos nuevos (eje der.)")
    ax2.axhline(y=threshold_data, color="orange", linestyle="--", linewidth=1.2,
                alpha=0.8, label=f"Umbral datos ({threshold_data}%)")
    ax2.set_ylabel("% datos nuevos desde último índice", color="orange")
    ax2.set_ylim(0, max(max(pct_news) * 1.5, threshold_data * 2))
    ax2.tick_params(axis="y", labelcolor="orange")

    # Marcadores de decisión
    rebuild_d = [d for d, dec in zip(dates, decisions) if dec == "REBUILD"]
    rebuild_c = [c for c, dec in zip(cosines, decisions) if dec == "REBUILD"]
    if rebuild_d:
        ax.scatter(rebuild_d, rebuild_c, color="red", s=160, zorder=5,
                   marker="X", label="Decisión REBUILD")

    keep_d = [d for d, dec in zip(dates, decisions) if dec == "KEEP"]
    keep_c = [c for c, dec in zip(cosines, decisions) if dec == "KEEP"]
    if keep_d:
        ax.scatter(keep_d, keep_c, color="green", s=90, zorder=5,
                   label="Decisión KEEP")

    ax.set_xlabel("Fecha del chequeo (UTC)")
    ax.set_title("GATOBYTE — Monitoreo de Drift: Calidad del Índice RAG")
    ax.grid(True, alpha=0.25)

    # Leyenda combinada de ambos ejes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
              fontsize=8, ncol=2)

    if len(dates) > 1:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        fig.autofmt_xdate()
    else:
        ax.set_xticks(dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        fig.autofmt_xdate()

    plt.tight_layout()
    chart_path = Path("reports/drift_trend.png")
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"[✓] Gráfica de tendencia → reports/drift_trend.png  "
          f"({len(history)} punto(s))")
    return str(chart_path)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*62)
    print("  GATOBYTE · Política de Actualización · CRISP-DM: Deployment")
    print("="*62 + "\n")

    cfg      = load_config()
    manifest = load_manifest(cfg)

    if manifest is None:
        print("[→] Acción requerida: python src/01_build_index.py")
    else:
        print(f"  Versión activa   : {manifest['version']}")
        print(f"  Chunks indexados : {manifest['num_chunks']:,}")
        print(f"  Modelo           : {manifest['model_name']}")
        print(f"  Chunk overlap    : {manifest.get('chunk_overlap', 'N/A')} chars\n")

        # ── Evaluar los 3 triggers ────────────────────────
        print("[→] Verificando triggers...")
        age_trigger,   age_days = check_age(manifest, cfg)
        drift_trigger, avg_sim  = check_quality_drift(manifest, cfg)
        data_trigger,  pct_new  = check_new_data(manifest, cfg)

        lim_age  = cfg["update_policy"]["max_age_days"]
        lim_cos  = cfg["update_policy"]["min_avg_cosine_sim"]
        lim_data = cfg["update_policy"]["new_data_threshold_pct"]

        print(f"\n  [T1] Antigüedad     : {age_days:.1f} días  "
              f"(límite: {lim_age}d)  "
              f"→ {'⚠  TRIGGER' if age_trigger else '✅ OK'}")
        print(f"  [T2] Similitud ctrl : {avg_sim:.4f}  "
              f"(mínimo: {lim_cos})  "
              f"→ {'⚠  DRIFT'   if drift_trigger else '✅ OK'}")
        print(f"  [T3] Datos nuevos   : {pct_new:.1f}%  "
              f"(umbral: {lim_data}%)  "
              f"→ {'⚠  TRIGGER' if data_trigger else '✅ OK'}")

        # ── Decisión final ────────────────────────────────
        reasons      = []
        needs_update = False
        if age_trigger  : needs_update = True; reasons.append(f"age>{lim_age}d")
        if drift_trigger: needs_update = True; reasons.append(f"cosine<{lim_cos}")
        if data_trigger : needs_update = True; reasons.append(f"new_data>{lim_data}%")

        print()
        if needs_update:
            print(f"  ⚠  DECISIÓN: RE-INDEXAR")
            print(f"     Triggers: {', '.join(reasons)}")
            print(f"     Estrategia: {cfg['update_policy']['rebuild_strategy'].upper()}")
            print(f"     → Ejecutar: python src/01_build_index.py")
        else:
            print(f"  ✅ DECISIÓN: MANTENER el índice actual")

        # ── Limpieza de versiones antiguas ────────────────
        print()
        cleanup_old_versions(cfg)

        # ── Registrar en MLflow ───────────────────────────
        log_policy_decision(
            manifest, age_days, avg_sim, pct_new,
            needs_update, reasons, cfg
        )

        # ── Historial persistente + gráfica de tendencia ─
        save_monitoring_history(
            manifest, age_days, avg_sim, pct_new,
            needs_update, reasons
        )
        plot_drift_trend(cfg)

        # ── Resumen de la política (para el informe) ──────
        print("\n" + "─"*62)
        print("  POLÍTICA DE ACTUALIZACIÓN DOCUMENTADA")
        print("─"*62)
        policy = cfg["update_policy"]
        print(f"  Antigüedad máxima       : {policy['max_age_days']} días")
        print(f"  Similitud coseno mínima : {policy['min_avg_cosine_sim']}")
        print(f"  Umbral datos nuevos     : {policy['new_data_threshold_pct']}%")
        print(f"  Estrategia de rebuild   : {policy['rebuild_strategy'].upper()}")
        print(f"  Versiones a retener     : {cfg['faiss']['keep_n_versions']}")
        print()
        print("  FLUJO MLOPS (CRISP-DM):")
        print("  ┌─ 01_build_index.py        Data Prep + Modeling + versionado")
        print("  ├─ 02_evaluate_retrieval.py Evaluation + P@K + MRR + /reports/")
        print("  ├─ 03_update_policy.py      Deployment + 3 triggers KEEP/REBUILD")
        print("  └─ demo/app.py              Deployment → interfaz Streamlit")
        print()
        print(f"  MLflow UI: mlflow ui  →  experimento: "
              f"'{cfg['mlflow']['experiment_name']}'")
