# GATOBYTE — Análisis de Reseñas de Amazon

> **Grupo:** GATOBYTE · UCB San Pablo · Machine Learning · 2026
> **Integrantes:** Ivonne Colque · Dilan Mamani · Tania Pérez · Ignacio Retamozo · Adriana Rocha

## Descripción del Proyecto

Este proyecto construye un sistema de análisis de reseñas de productos electrónicos de Amazon con dos componentes principales:

| Módulo | Tarea | Tecnología |
|---|---|---|
| Clasificación de Sentimiento | Multiclase (Positivo / Neutral / Negativo) | LightGBM + TF-IDF · DistilBERT + LoRA |
| Búsqueda Semántica (RAG) | Recuperación por similitud semántica | FAISS · MiniLM · embeddings multilingüe |

La metodología adoptada es **CRISP-DM**, con tracking de experimentos en MLflow, versionado automático del índice vectorial y una demo web interactiva con múltiples vistas.

---

## Estructura del Repositorio

```
amazon-reviews-gatobyte/
│
├── data/
│   ├── sample_ml.parquet                  # Dataset principal (50k reseñas)
│   ├── pipeline_transformacion_cpu.joblib # Pipeline de features del baseline
│   ├── lightgbm_tuned_final_cpu.joblib    # Modelo LightGBM entrenado
│   ├── label_encoder.joblib               # Encoder de clases
│   ├── metadata_modelo_final.json         # Métricas y config del baseline
│   ├── metadata_transformer.json          # Métricas del transformer
│   ├── metricas_todos_modelos.json        # Tabla comparativa de modelos
│   ├── umap_coords.csv                    # Coordenadas 2D para visualización
│   └── FUENTE.md                          # Procedencia del dataset
│
├── models/
│   ├── distilbert_lora/                   # Adaptador LoRA entrenado
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── tokenizer.json / tokenizer_config.json
│   └── faiss_index/
│       ├── latest.json                    # Puntero a la versión activa
│       └── v_YYYYMMDD_HASH/
│           ├── index.faiss
│           ├── chunks_metadata.pkl
│           └── manifest.json
│
├── src/
│   ├── 01_build_index.py          # Construye el índice FAISS + registro MLflow
│   ├── 02_evaluate_retrieval.py   # Precision@K, MRR, latencia del RAG
│   ├── 03_update_policy.py        # Política de rebuild del índice (3 triggers)
│   └── 04_baseline_comparison.py  # TF-IDF vs MiniLM embeddings
│
├── demo/
│   ├── main.py                    # API FastAPI (entry point)
│   ├── routes/
│   │   ├── sentiment.py           # POST /api/predict
│   │   ├── embeddings.py          # GET  /api/embeddings/umap
│   │   ├── metrics.py             # GET  /api/metrics
│   │   └── rag.py                 # GET  /api/search · /api/info
│   └── static/
│       ├── index.html · style.css
│       ├── app.js        # Tab Buscar (RAG)
│       ├── sentiment.js  # Tab Clasificador
│       ├── umap.js       # Tab Embeddings
│       └── dashboard.js  # Tab Métricas
│
├── config/
│   └── rag_config.yaml            # Configuración central del pipeline
├── reports/                       # Reportes JSON/TXT de evaluación
├── mlruns/                        # Artefactos MLflow (generado automáticamente)
├── requirements.txt
└── README.md
```

---

## Dataset

**Amazon Reviews 2023** — subconjunto de Electronics
McAuley Lab, UC San Diego · Hou et al., 2024 (arXiv:2403.03952)

| Atributo | Detalle |
|---|---|
| Registros | 50,000 reseñas (muestreo aleatorio, `random_seed=42`) |
| Chunks generados | 61,408 fragmentos (`chunk_size=500`, `overlap=100`) |
| Dimensión de embeddings | 384 (paraphrase-multilingual-MiniLM-L12-v2) |
| Variable objetivo | Sentimiento derivado del rating (1–2★ Neg · 3★ Neu · 4–5★ Pos) |

Ver [data/FUENTE.md](data/FUENTE.md) para detalles de procedencia y columnas.

> **Data Leakage Prevention:** La variable `rating` fue excluida del entrenamiento de sentimiento al ser la base lógica del target.

---

## Instalación y Reproducción

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd amazon-reviews-gatobyte
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
venv/bin/python -m pip install -r requirements.txt
```

### 3. Construir el índice vectorial

```bash
python src/01_build_index.py
```

Carga el parquet, aplica chunking, genera embeddings con MiniLM y construye el índice FAISS. Registra el run en MLflow.

### 4. Ejecutar la demo

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
uvicorn demo.main:app --port 7860
```

Abrir [http://localhost:7860](http://localhost:7860).

> **Nota macOS ARM (Apple Silicon):** los cuatro `export` son necesarios para evitar un conflicto de OpenMP entre LightGBM y PyTorch, y para que los modelos HuggingFace carguen desde caché local.

---

## Demo Web

| Tab | Descripción |
|---|---|
| **Buscar** | Búsqueda semántica sobre reseñas en español o inglés (RAG + FAISS) |
| **Clasificador** | Clasifica texto como Positivo / Neutral / Negativo (baseline y transformer) |
| **Embeddings** | Visualización UMAP del espacio semántico (384d → 2d) |
| **Métricas** | Dashboard de métricas de entrenamiento de ambos modelos |
| **Rendimiento** | Evaluación de recuperación del RAG (Precision@K, MRR, latencia) |
| **Explorar** | Explorador del dataset con filtros por sentimiento y categoría |
| **Sistema** | Estado del índice RAG activo y configuración del pipeline |

### Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/predict` | Body: `{"text": "...", "model": "baseline" \| "transformer"}` |
| `GET` | `/api/metrics` | Métricas de ambos modelos |
| `GET` | `/api/embeddings/umap` | Coordenadas UMAP |
| `GET` | `/api/search?q=...&top_k=5` | Búsqueda semántica RAG |
| `GET` | `/api/info` | Info del índice FAISS activo |

---

## Metodología

El proyecto sigue las fases de **CRISP-DM**:

```
Business Understanding → Data Understanding → Data Preparation
        ↓
   Modeling → Evaluation → Deployment
```

| Fase | Artefacto |
|---|---|
| 1. Business Understanding | Definición de módulos y métricas objetivo |
| 2. Data Understanding | EDA de distribución de texto, sentimiento y categorías |
| 3. Data Preparation | Limpieza, chunking con overlap, imputación de precio |
| 4. Modeling | LightGBM + TF-IDF · DistilBERT+LoRA · FAISS + MiniLM |
| 5. Evaluation | Métricas de clasificación + recuperación RAG |
| 6. Deployment | Demo FastAPI + política de actualización del índice |

---

## Resultados

### Módulo I — Clasificación de Sentimiento

#### Baseline — LightGBM + TF-IDF (CPU)

Pipeline clásico con TF-IDF (10,000 features), features tabulares (longitud, precio, categoría) y optimización con Optuna.

| Métrica | Validación | Test |
|---|---|---|
| F1 Macro | 0.6953 | 0.6945 |
| F1 Weighted | 0.8430 | 0.8422 |
| Balanced Accuracy | 0.7578 | 0.7575 |
| ROC-AUC | 0.9327 | 0.9320 |

#### Transformer — DistilBERT + LoRA (PEFT)

Fine-tuning eficiente con adaptadores LoRA (r=8, α=16) sobre capas de atención. Modelo ganador del proyecto.

| Métrica | Test |
|---|---|
| F1 Macro | 0.7272 |
| Balanced Accuracy | 0.7833 |
| ROC-AUC | 0.9483 |
| PR-AUC | 0.7762 |
| Score Compuesto | 0.8104 |

#### Comparativa de todos los modelos evaluados

| Modelo | Representación | F1 Macro | Bal. Acc | ROC-AUC | Score |
|---|---|---|---|---|---|
| **DistilBERT + LoRA ★** | Contextual | **0.727** | **0.783** | **0.948** | **0.810** |
| Stacking Ensemble | TF-IDF + Emb + BERT | 0.725 | 0.781 | 0.947 | 0.808 |
| LightGBM TF-IDF | TF-IDF + Tabular | 0.695 | 0.758 | 0.932 | 0.785 |
| XGBoost TF-IDF | TF-IDF + Tabular | 0.679 | 0.749 | 0.931 | 0.776 |
| LogReg TF-IDF | TF-IDF + Tabular | 0.691 | 0.720 | 0.943 | 0.775 |
| LogReg Emb | MiniLM-384 | 0.648 | 0.710 | 0.905 | 0.744 |
| LightGBM Emb | MiniLM-384 | 0.594 | 0.696 | 0.901 | 0.717 |
| Naive Bayes | TF-IDF | 0.557 | 0.557 | 0.900 | 0.660 |

### Módulo II — Búsqueda Semántica (RAG)

Evaluación sobre 7 queries en español con corpus en inglés (búsqueda cross-language):

| Query | Precision@5 | MRR | Cosine Sim | Latencia |
|---|---|---|---|---|
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

## MLOps y Tracking

### Pipeline reproducible

Cuatro scripts independientes y configurables desde `config/rag_config.yaml`:

```bash
python src/01_build_index.py          # Construir índice FAISS + registrar run MLflow
python src/02_evaluate_retrieval.py   # Evaluar calidad de recuperación
python src/03_update_policy.py        # Verificar política de actualización
python src/04_baseline_comparison.py  # Comparar TF-IDF vs embeddings MiniLM
```

### Tracking con MLflow

```bash
mlflow ui
# Abrir http://localhost:5000
# Experimento: "RAG_Electronics_Reviews"
```

Cada build del índice registra: parámetros (chunk_size, overlap, modelo de embeddings), métricas (Precision@K, MRR, latencia) y decisiones de la política de actualización.

### Versionado del índice

El directorio `models/faiss_index/` mantiene múltiples versiones. `latest.json` apunta a la versión activa, desacoplando el código de una versión específica. Se retienen las últimas 3 versiones.

### Política de actualización automática

| Trigger | Condición | Acción |
|---|---|---|
| T1 — Antigüedad | índice > 30 días | REBUILD |
| T2 — Drift de calidad | cosine < 0.30 en queries de control | REBUILD |
| T3 — Volumen de datos nuevos | > 10% del tamaño actual | REBUILD |

Cada decisión (KEEP / REBUILD) queda registrada en MLflow con los valores exactos de cada trigger.

---

## Stack Tecnológico

| Librería | Uso |
|---|---|
| `FastAPI` / `uvicorn` | API REST y servidor de la demo |
| `LightGBM` | Modelo baseline de clasificación |
| `transformers` + `peft` | DistilBERT + adaptadores LoRA |
| `sentence-transformers` | Embeddings multilingüe para RAG |
| `faiss-cpu` | Índice vectorial para búsqueda semántica |
| `MLflow` | Tracking de experimentos y versionado |
| `scikit-learn` | Pipelines, TF-IDF, métricas |
| `Optuna` | Optimización de hiperparámetros |
| `Plotly` | Visualizaciones interactivas (UMAP, dashboard) |
| `Polars` / `pandas` | Manipulación de datos |

---

## Integrantes

| Nombre 
|---|---|
| Adriana Nathalie Rocha Vedia |
| Ivonne Micaela Colque Murillo |
| Dilan Obed Mamani Pamuri | 
| Tania Morelia Pérez Dick | 
| Ignacio Retamozo Torrez |

<div align="center">
  <sub>Proyecto académico · Machine Learning · UCB San Pablo · 2026</sub>
</div>
