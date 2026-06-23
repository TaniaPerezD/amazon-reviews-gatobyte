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
# T3 — Volumen de datos nuevos
# ─────────────────────────────────────────────────────────
def check_new_data(manifest: dict, cfg: dict,
                   new_records_count: int = 0) -> tuple[bool, float]:
    """
    Si los datos nuevos desde la última indexación superan el umbral %,
    se activa el trigger de re-indexación.
    En producción, new_records_count vendría de un contador en la fuente.
    """
    current_size = manifest.get("sample_size", 1)
    threshold    = cfg["update_policy"]["new_data_threshold_pct"]
    pct_new      = (new_records_count / current_size) * 100
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
        data_trigger,  pct_new  = check_new_data(manifest, cfg,
                                                  new_records_count=0)

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
