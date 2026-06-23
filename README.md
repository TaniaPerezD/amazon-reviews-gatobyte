# GATOBYTE — Mini Proyecto MLOps + RAG
## Recuperación Semántica en Reseñas de Electrónica
**UCB San Pablo · Machine Learning · Equipo: GATOBYTE**

---

## Descripción

Sistema de recuperación semántica (RAG) sobre reseñas de productos electrónicos de Amazon. Dada una consulta en lenguaje natural —en español o inglés— el sistema devuelve los fragmentos de reseña más relevantes usando embeddings multilingües y búsqueda vectorial con FAISS.

El pipeline sigue la metodología CRISP-DM e incluye tracking con MLflow, versionado del índice, política de actualización automática y una demo web interactiva.

---

## Estructura del repositorio

```
├── data/
│   ├── sample_ml.parquet          ← dataset
│   └── FUENTE.md                  ← procedencia y descripción del dataset
├── src/
│   ├── 01_build_index.py          ← Data Prep + Embeddings + FAISS + MLflow
│   ├── 02_evaluate_retrieval.py   ← Precision@K, MRR, Cosine Sim, Latencia
│   ├── 03_update_policy.py        ← Triggers de rebuild + MLflow
│   └── 04_baseline_comparison.py  ← Comparación TF-IDF vs MiniLM
├── demo/
│   ├── main.py                    ← API FastAPI (búsqueda, chunks, eval)
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── models/
│   └── faiss_index/
│       ├── latest.json            ← puntero a la versión activa
│       └── v_YYYYMMDD_HASH/
│           ├── index.faiss
│           ├── chunks_metadata.pkl
│           └── manifest.json
├── reports/
│   ├── retrieval_eval_report.json
│   ├── retrieval_eval_summary.txt
│   ├── baseline_comparison_report.json
│   └── baseline_comparison_summary.txt
├── notebooks/
│   └── rag_crisp_dm_exploratorio.ipynb
├── config/
│   └── rag_config.yaml            ← configuración central del pipeline
├── mlruns/                        ← artefactos MLflow (generado automáticamente)
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución del pipeline

### Paso 1 — Construir el índice vectorial
```bash
python src/01_build_index.py
```
Carga el parquet, aplica chunking (500 chars, overlap 100), genera embeddings con `paraphrase-multilingual-MiniLM-L12-v2`, construye el índice FAISS y registra el run en MLflow.

### Paso 2 — Evaluar calidad de recuperación
```bash
python src/02_evaluate_retrieval.py
```
Ejecuta las queries de evaluación definidas en `config/rag_config.yaml` y guarda Precision@K, MRR, similitud coseno y latencia en `reports/`.

### Paso 3 — Verificar política de actualización
```bash
python src/03_update_policy.py
```
Evalúa los tres triggers (antigüedad, drift, volumen) y registra la decisión KEEP / REBUILD en MLflow.

### Paso 4 — Comparación con baseline TF-IDF
```bash
python src/04_baseline_comparison.py
```
Compara TF-IDF clásico vs embeddings MiniLM sobre las mismas queries. Resultados en `reports/baseline_comparison_summary.txt`.

### Paso 5 — Demo interactiva
```bash
uvicorn demo.main:app --reload --port 7860
# Abrir http://localhost:7860
```

### Ver runs en MLflow
```bash
mlflow ui
# Abrir http://localhost:5000
# Experimento: "RAG_Electronics_Reviews"
```

---

## Metodología CRISP-DM

| Fase | Script / Artefacto | Descripción |
|------|--------------------|-------------|
| 1. Business Understanding | `notebooks/rag_crisp_dm_exploratorio.ipynb` | Definición del problema RAG |
| 2. Data Understanding | `notebooks/` → sección EDA | Distribución de texto, sentiment, categorías |
| 3. Data Preparation | `src/01_build_index.py` → `load_and_prepare`, `chunk_reviews` | Carga parquet, filtrado, chunking con overlap |
| 4. Modeling | `src/01_build_index.py` → `generate_embeddings`, `build_faiss_index` | Embeddings MiniLM + índice FAISS IndexFlatIP |
| 5. Evaluation | `src/02_evaluate_retrieval.py` + `src/04_baseline_comparison.py` | Métricas de recuperación + comparación baseline |
| 6. Deployment | `demo/main.py` + `src/03_update_policy.py` | Demo FastAPI + política de actualización |

---

## Métricas de evaluación

Resultados sobre 7 queries de prueba en español (reseñas en inglés — cross-language):

| Query | Precision@5 | MRR | Cosine Sim | Latencia |
|-------|-------------|-----|------------|----------|
| problemas frecuentes con el producto | 0.40 | 0.50 | 0.663 | 20 ms |
| opiniones sobre duración de batería | 1.00 | 1.00 | 0.794 | 46 ms |
| defectos de fabricación y materiales | 0.80 | 1.00 | 0.653 | 19 ms |
| facilidad de configuración y uso | 0.60 | 1.00 | 0.745 | 19 ms |
| ruido y temperatura del dispositivo | 1.00 | 1.00 | 0.630 | 18 ms |
| relación calidad precio | 0.80 | 1.00 | 0.774 | 18 ms |
| pantalla y calidad de imagen | 1.00 | 1.00 | 0.725 | 17 ms |
| **MEDIA** | **0.80** | **0.93** | — | **22 ms** |

Todos los objetivos superados: Precision@K ≥ 0.60  · MRR ≥ 0.60  · Latencia < 200 ms 

---

## Comparación con baseline TF-IDF

| Query (ES) | TF-IDF P@K | MiniLM P@K | Ganador |
|------------|-----------|------------|---------|
| problemas frecuentes | 0.80 | 0.40 | TF-IDF |
| opiniones sobre batería | 1.00 | 1.00 | MiniLM |
| defectos de fabricación | 1.00 | 0.20 | TF-IDF |
| facilidad de uso | 0.20 | 0.20 | MiniLM |
| ruido excesivo | 0.80 | 0.80 | MiniLM |
| **MEDIA** | **0.76** | **0.52** | — |

**Conclusión:** TF-IDF supera a MiniLM en precisión bruta cuando las keywords de la query coinciden léxicamente con el corpus en inglés. MiniLM supera a TF-IDF en queries donde la semántica importa más que las palabras exactas, y es el único enfoque viable para búsqueda cross-language (query en español, corpus en inglés) sin traducción previa. La comparación justifica el uso del modelo multilingüe para el caso de uso real del sistema.

---

## Política de actualización del índice

| Trigger | Condición | Acción |
|---------|-----------|--------|
| T1 — Antigüedad | índice > 30 días | REBUILD |
| T2 — Drift de calidad | cosine < 0.30 en queries de control | REBUILD |
| T3 — Volumen de datos nuevos | > 10% del tamaño actual | REBUILD |

- Estrategia: **FULL REBUILD**
- Retención: **últimas 3 versiones**
- Cada decisión queda registrada en MLflow con los valores exactos de cada trigger

---

## Dataset

**Amazon Reviews 2023** — subconjunto de Electronics  
McAuley Lab, UC San Diego · Hou et al., 2024 (arXiv:2403.03952)

- 50 000 (para el desarollo del rag) reseñas muestreadas aleatoriamente (`random_seed=42`) 
- 61 408 chunks generados (chunk_size=500, overlap=100)
- Embeddings de dimensión 384

Ver `data/FUENTE.md` para detalles completos de procedencia y columnas.

