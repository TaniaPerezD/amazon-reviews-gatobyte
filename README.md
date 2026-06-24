# GATOBYTE — Análisis de Reseñas de Amazon
## Clasificación de Sentimiento + Búsqueda Semántica (RAG)
**UCB San Pablo · Machine Learning · Equipo: GATOBYTE**

---

## Descripción

Sistema de análisis de reseñas de productos electrónicos de Amazon con dos componentes principales:

- **Clasificador de sentimiento**: predice si una reseña es positiva, neutral o negativa. Incluye un modelo baseline (LightGBM + TF-IDF) y un modelo transformer (DistilBERT + LoRA).
- **Búsqueda semántica (RAG)**: dado un texto en español o inglés, recupera los fragmentos de reseñas más relevantes usando embeddings multilingües y un índice FAISS.

El proyecto sigue la metodología CRISP-DM e incluye tracking de experimentos con MLflow, versionado del índice RAG, política de actualización automática y una demo web interactiva con múltiples vistas.

---

## Estructura del repositorio

```
├── data/
│   ├── sample_ml.parquet                  ← dataset 
│   ├── pipeline_transformacion_cpu.joblib ← pipeline de features para baseline
│   ├── lightgbm_tuned_final_cpu.joblib    ← modelo LightGBM entrenado
│   ├── label_encoder.joblib               ← encoder de clases (neg/neu/pos)
│   ├── metadata_modelo_final.json         ← métricas y config del baseline
│   ├── metadata_transformer.json          ← métricas del transformer
│   ├── metricas_todos_modelos.json        ← tabla comparativa de todos los modelos
│   ├── metricas_referencia.json           ← métricas de referencia para drift
│   ├── umap_coords.csv                    ← coordenadas 2D para visualización
│   └── FUENTE.md                          ← procedencia y descripción del dataset
├── models/
│   ├── distilbert_lora/                   ← adaptador LoRA entrenado
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── faiss_index/
│       ├── latest.json                    ← puntero a la versión activa
│       └── v_YYYYMMDD_HASH/
│           ├── index.faiss
│           ├── chunks_metadata.pkl
│           └── manifest.json
├── src/
│   ├── 01_build_index.py          ← construye el índice FAISS + registro MLflow
│   ├── 02_evaluate_retrieval.py   ← Precision@K, MRR, latencia del RAG
│   ├── 03_update_policy.py        ← política de rebuild del índice
│   └── 04_baseline_comparison.py  ← TF-IDF vs MiniLM
├── demo/
│   ├── main.py                    ← API FastAPI (entry point)
│   └── routes/
│       ├── sentiment.py           ← /api/predict (baseline + transformer)
│       ├── embeddings.py          ← /api/embeddings/umap
│       ├── metrics.py             ← /api/metrics
│       ├── rag.py                 ← /api/search, /api/info
│       ├── inferencia_cpu.py      ← lógica de inferencia del baseline
│       └── migrar_pipeline_cpu.py ← clases del pipeline CPU (necesarias para joblib)
│   └── static/
│       ├── index.html
│       ├── app.js        ← tab Buscar (RAG)
│       ├── sentiment.js  ← tab Clasificador
│       ├── umap.js       ← tab Embeddings
│       ├── dashboard.js  ← tab Métricas
│       └── style.css
├── notebooks/
│   └── rag_crisp_dm_exploratorio.ipynb
├── reports/
│   ├── retrieval_eval_report.json
│   ├── retrieval_eval_summary.txt
│   ├── baseline_comparison_report.json
│   ├── baseline_comparison_summary.txt
│   ├── monitoring_history.json
│   └── drift_trend.png
├── config/
│   └── rag_config.yaml            ← configuración central del pipeline
├── mlruns/                        ← artefactos MLflow (generado automáticamente)
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
# 1. Crear entorno virtual (Python 3.10)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecutar la demo

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
uvicorn demo.main:app --port 7860
```

Luego abrir [http://localhost:7860](http://localhost:7860).

> **Nota macOS ARM (Apple Silicon):** los cuatro `export` son necesarios para evitar un conflicto de OpenMP entre LightGBM y PyTorch, y para que los modelos HuggingFace carguen desde caché local sin intentar conectarse a internet.

### Tabs de la demo

| Tab | Descripción |
|-----|-------------|
| **Buscar** | Búsqueda semántica sobre reseñas (RAG). Requiere índice FAISS generado. |
| **Clasificador** | Clasifica un texto como positivo / neutral / negativo. Soporta modelo baseline y transformer. |
| **Embeddings** | Visualización UMAP del espacio semántico de los embeddings. |
| **Métricas** | Dashboard con métricas de entrenamiento de ambos modelos. |
| **Rendimiento** | Tabla de evaluación de recuperación del RAG (Precision@K, MRR, latencia). |
| **Explorar** | Explorador del dataset de reseñas. |
| **Sistema** | Estado del índice RAG y configuración activa. |

---

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/api/predict` | Clasifica sentimiento. Body: `{"text": "...", "model": "baseline" \| "transformer", "price": 0.0}` |
| `GET` | `/api/metrics` | Métricas de entrenamiento de ambos modelos |
| `GET` | `/api/embeddings/umap` | Coordenadas UMAP de los embeddings |
| `GET` | `/api/search?q=...&top_k=5` | Búsqueda semántica RAG |
| `GET` | `/api/info` | Información del índice FAISS activo |

---

## Modelos de clasificación de sentimiento

### Baseline — LightGBM + TF-IDF (CPU)

Pipeline clásico con features de texto (TF-IDF 10 000 features) y features tabulares (longitud, precio, categoría). Optimizado con Optuna.

| Métrica | Validación | Test |
|---------|-----------|------|
| F1 Macro | 0.6953 | 0.6945 |
| F1 Weighted | 0.8430 | 0.8422 |
| Balanced Accuracy | 0.7578 | 0.7575 |
| ROC-AUC | 0.9327 | 0.9320 |

### Transformer — DistilBERT + LoRA (PEFT)

Fine-tuning eficiente de DistilBERT-base-uncased con adaptadores LoRA (r=8, α=16) sobre las capas de atención. Modelo ganador del proyecto.

| Métrica | Test |
|---------|------|
| F1 Macro | 0.7272 |
| Balanced Accuracy | 0.7833 |
| ROC-AUC | 0.9483 |
| PR-AUC | 0.7762 |
| Score Compuesto | 0.8104 |

### Comparativa de todos los modelos evaluados

| Modelo | Representación | F1 Macro | Bal. Acc | ROC-AUC | Score |
|--------|---------------|---------|---------|---------|-------|
| **DistilBERT+LoRA** | Contextual | **0.727** | **0.783** | **0.948** | **0.810** |
| Stacking Ensemble | TF-IDF+Emb+BERT | 0.725 | 0.781 | 0.947 | 0.808 |
| LightGBM TF-IDF | TF-IDF + Tabular | 0.695 | 0.758 | 0.932 | 0.785 |
| XGBoost TF-IDF | TF-IDF + Tabular | 0.679 | 0.749 | 0.931 | 0.776 |
| LogReg TF-IDF | TF-IDF + Tabular | 0.691 | 0.720 | 0.943 | 0.775 |
| LogReg Emb | MiniLM-384 | 0.648 | 0.710 | 0.905 | 0.744 |
| LightGBM Emb | MiniLM-384 | 0.594 | 0.696 | 0.901 | 0.717 |
| Naive Bayes | TF-IDF | 0.557 | 0.557 | 0.900 | 0.660 |

---

## Pipeline RAG

### Construir el índice vectorial
```bash
python src/01_build_index.py
```
Carga el parquet, aplica chunking (500 chars, overlap 100), genera embeddings con `paraphrase-multilingual-MiniLM-L12-v2` (dim 384), construye el índice FAISS y registra el run en MLflow.

### Evaluar calidad de recuperación
```bash
python src/02_evaluate_retrieval.py
```
Ejecuta las queries de evaluación de `config/rag_config.yaml` y guarda Precision@K, MRR, similitud coseno y latencia en `reports/`.

### Verificar política de actualización
```bash
python src/03_update_policy.py
```
Evalúa tres triggers (antigüedad, drift de calidad, volumen de datos nuevos) y registra la decisión KEEP / REBUILD en MLflow.

### Comparación con baseline TF-IDF
```bash
python src/04_baseline_comparison.py
```
Compara TF-IDF clásico vs embeddings MiniLM sobre las mismas queries. Resultados en `reports/baseline_comparison_summary.txt`.

---

## Métricas de recuperación RAG

Evaluación sobre 7 queries en español (corpus en inglés — búsqueda cross-language):

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

Objetivos cumplidos: Precision@K ≥ 0.60 · MRR ≥ 0.60 · Latencia < 200 ms

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

## Metodología CRISP-DM

| Fase | Artefacto | Descripción |
|------|-----------|-------------|
| 1. Business Understanding | `notebooks/rag_crisp_dm_exploratorio.ipynb` | Definición del problema |
| 2. Data Understanding | `notebooks/` → sección EDA | Distribución de texto, sentimiento, categorías |
| 3. Data Preparation | `src/01_build_index.py` | Carga parquet, filtrado, chunking con overlap |
| 4. Modeling | `src/01_build_index.py` + `models/distilbert_lora/` | Embeddings MiniLM + FAISS + LoRA fine-tuning |
| 5. Evaluation | `src/02_evaluate_retrieval.py` + `src/04_baseline_comparison.py` | Métricas de recuperación y clasificación |
| 6. Deployment | `demo/` + `src/03_update_policy.py` | Demo FastAPI + política de actualización |

---

## Tracking con MLflow

```bash
mlflow ui
# Abrir http://localhost:5000
# Experimento: "RAG_Electronics_Reviews"
```

Registra: embeddings generados, parámetros del índice, métricas de evaluación y decisiones de la política de actualización.

---

## Dataset

**Amazon Reviews 2023** — subconjunto de Electronics  
McAuley Lab, UC San Diego · Hou et al., 2024 (arXiv:2403.03952)

- 50 000 reseñas muestreadas aleatoriamente (`random_seed=42`)
- 61 408 chunks generados (chunk_size=500, overlap=100)
- Embeddings de dimensión 384

Ver [data/FUENTE.md](data/FUENTE.md) para detalles completos de procedencia y columnas.
