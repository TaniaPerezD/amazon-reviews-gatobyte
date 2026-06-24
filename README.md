# GATOBYTE — Analisis de Resenas de Amazon Electronics

**UCB San Pablo · Machine Learning 2026-I · Grupo 1**  
GATOBYTE · Ivonne Colque · Dilan Mamani · Tania Pérez · Ignacio Retamozo · Adriana Rocha

Proyecto integrador de la materia de Machine Learning desarrollado en tres fases articuladas sobre el dataset Amazon Reviews 2023, categoria Electronics. La metodologia seguida es CRISP-DM y el trabajo cubre aprendizaje supervisado, no supervisado, deep learning, embeddings y un pipeline MLOps con recuperacion semantica.

---

## Estructura del repositorio

```
amazon-reviews-gatobyte/
├── config/
│   └── rag_config.yaml                        <- configuracion central del pipeline RAG
├── data/
│   ├── FUENTE.md                              <- procedencia y descripcion del dataset
│   ├── label_encoder.joblib                   <- encoder de clases del modelo baseline
│   ├── lightgbm_tuned_final_cpu.joblib        <- modelo LightGBM entrenado (CPU)
│   ├── metadata_modelo_final.json             <- metricas y parametros del baseline
│   ├── metadata_transformer.json             <- metricas del DistilBERT + LoRA
│   ├── metricas_todos_modelos.json            <- tabla comparativa completa (8 modelos)
│   ├── pipeline_transformacion_cpu.joblib     <- pipeline de preprocesamiento (CPU)
│   └── umap_coords.csv                        <- coordenadas UMAP de embeddings (5002 puntos)
├── demo/
│   ├── main.py                                <- servidor FastAPI principal
│   ├── routes/
│   │   ├── embeddings.py                      <- endpoint /api/embeddings/umap
│   │   ├── inferencia_cpu.py                  <- funcion de inferencia para produccion
│   │   ├── metrics.py                         <- endpoint /api/metrics
│   │   ├── migrar_pipeline_cpu.py             <- clases CPU del pipeline migrado
│   │   ├── rag.py                             <- endpoints de busqueda semantica FAISS
│   │   └── sentiment.py                       <- endpoints /api/predict (baseline + transformer)
│   └── static/
│       ├── app.js                             <- logica principal del frontend
│       ├── dashboard.js                       <- tab de metricas comparativas
│       ├── index.html                         <- interfaz web
│       ├── sentiment.js                       <- tab del clasificador de sentimiento
│       ├── style.css                          <- estilos principales
│       └── umap.js                            <- tab de visualizacion de embeddings
├── models/
│   ├── distilbert_lora/                       <- adaptador LoRA del transformer
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── distilbert_lora_config.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   └── faiss_index/
│       └── latest.json                        <- puntero a la version activa del indice
├── notebooks/
│   ├── amazon_full_pipeline.ipynb             <- preprocesamiento general (Fase 2)
│   ├── Analisis_Sentimiento_Electronics_GATOBYTE.ipynb  <- modelos ML de sentimiento
│   ├── CLUSTERING_GATOBYTE_AUTOML_.ipynb      <- clustering y AutoML
│   ├── DEEP_LEARNING_GATOBYTE_EMBEDDING.ipynb <- DistilBERT, embeddings y UMAP
│   ├── Gatobyte Regression.ipynb              <- regresion de rating y helpfulness
│   ├── NB-02_Inferencia_Nuevos_Datos.ipynb    <- ejemplo de inferencia CPU
│   ├── NB_01c_Stacking_Analisis.ipynb         <- stacking y comparativa final de modelos
│   └── rag_crisp_dm_exploratorio.ipynb        <- exploracion del pipeline RAG
├── reports/
│   ├── baseline_comparison_report.json
│   ├── baseline_comparison_summary.txt
│   ├── retrieval_eval_report.json
│   └── retrieval_eval_summary.txt
├── src/
│   ├── 01_build_index.py                      <- construye el indice FAISS
│   ├── 02_evaluate_retrieval.py               <- evalua calidad de recuperacion
│   ├── 03_update_policy.py                    <- politica de actualizacion del indice
│   └── 04_baseline_comparison.py              <- comparacion TF-IDF vs MiniLM
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Dataset

**Amazon Reviews 2023** — categoria Electronics  
McAuley Lab, UC San Diego. Hou et al., 2024.  
Fuente: https://amazon-reviews-2023.github.io/

Se trabajo con dos subconjuntos segun la fase:

- Fase 2 (ML): 1,000,000 de resenas muestreadas aleatoriamente (`random_seed=42`), con 21 columnas y distribucion de sentimiento 74% positivo / 21% negativo / 7% neutro.
- Fase 4 (RAG): 50,000 resenas para construccion del indice vectorial. 61,408 chunks generados (chunk_size=500, overlap=100). Embeddings de dimension 384.

Ver `data/FUENTE.md` para detalles de columnas, criterio de muestreo y particion.

---

## Instalacion

Se requiere Python 3.12. Se recomienda crear un entorno virtual antes de instalar dependencias.

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

Torch debe instalarse por separado porque requiere una URL especifica para la version CPU:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

Luego el resto de dependencias:

```bash
pip install -r requirements.txt
```

El modelo base de DistilBERT se descarga de Hugging Face la primera vez (~268 MB). Para evitar que esto ocurra al arrancar el servidor, ejecutar esto una sola vez antes:

```bash
python -c "
from transformers import DistilBertForSequenceClassification
DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=3, ignore_mismatched_sizes=True
)
print('Modelo base cacheado correctamente')
"
```

---

## Ejecucion de la demo

```bash
python demo/main.py
```

El servidor arranca en http://localhost:7860. La primera vez que se inicia tarda unos segundos adicionales porque carga el pipeline CPU y el modelo DistilBERT en memoria.

Si el indice FAISS no existe todavia, la busqueda semantica no estara disponible pero el resto de la interfaz funciona normalmente. Para construir el indice:

```bash
python src/01_build_index.py
```

---

## Resumen de la demo interactiva

La interfaz tiene seis secciones accesibles desde el menu superior:

**Buscar** — recuperacion semantica sobre el indice FAISS. Se escribe una consulta en lenguaje natural (espanol o ingles) y el sistema devuelve los fragmentos de resena mas relevantes usando embeddings multilingues.

**Clasificador** — dado un texto de resena, predice si el sentimiento es positivo, negativo o neutral. Permite elegir entre el modelo baseline (LightGBM + TF-IDF) y el transformer (DistilBERT + LoRA). Muestra la etiqueta, el porcentaje de confianza y la distribucion de probabilidades para las tres clases.

**Embeddings** — visualizacion UMAP de 5,002 embeddings generados con all-MiniLM-L6-v2. Los puntos se pueden colorear por sentimiento, rating o categoria. Hacer clic en un punto muestra sus coordenadas y metadatos.

**Metricas** — dashboard comparativo con F1-Macro, F1-Weighted, Balanced Accuracy y ROC-AUC del baseline y el transformer. Incluye el anillo de F1-Macro y la tabla detallada con destacado del modelo ganador por metrica.

**Rendimiento** — resultados del pipeline de evaluacion de recuperacion semantica sobre 7 queries de prueba en espanol.

**Sistema** — estado del indice FAISS activo: version, numero de chunks, dimension de embeddings y politica de actualizacion.

---

## Resultados por fase

### Fase 2 — Machine Learning

Clasificacion de sentimiento (positivo / neutro / negativo) sobre 1M de resenas. Metrica principal: F1-Macro, para penalizar el sesgo hacia la clase mayoritaria (74% positivo).

| Modelo | F1-Macro | Bal. Accuracy | ROC-AUC |
|--------|----------|---------------|---------|
| Naive Bayes | 0.5574 | 0.5574 | 0.9000 |
| Logistic Regression TF-IDF | 0.6910 | 0.7200 | 0.9425 |
| XGBoost TF-IDF | 0.6790 | 0.7492 | 0.9310 |
| LightGBM TF-IDF (ganador) | 0.6945 | 0.7575 | 0.9320 |

El benchmark con FLAML AutoML (500 segundos, F1-Macro optimizado) confirmo LightGBM como mejor modelo. Los modelos manuales igualaron o superaron los resultados de AutoML.

Se implementaron ademas regresion de rating (CatBoost, MAE 0.29, R2 ~ 0.934) y clasificacion de helpfulness (LightGBM), y clustering de productos con PCA + KMeans k=3 (Silhouette 0.311).

### Fase 3 — Deep Learning y Embeddings

Se entrenaron embeddings con DistilBERT-base-uncased usando LoRA (PEFT) para clasificacion de sentimiento. La reduccion de dimensionalidad con UMAP permite visualizar la separacion semantica entre clases.

Se construyo ademas un ensemble por stacking combinando TF-IDF, embeddings MiniLM y DistilBERT.

Comparativa final (conjunto de test):

| Modelo | F1-Macro | Bal. Accuracy | ROC-AUC | Score Compuesto |
|--------|----------|---------------|---------|-----------------|
| DistilBERT + LoRA (ganador) | 0.7272 | 0.7833 | 0.9483 | 0.8104 |
| Stacking Ensemble | 0.7250 | 0.7811 | 0.9470 | 0.8084 |
| LightGBM TF-IDF | 0.6945 | 0.7575 | 0.9320 | 0.7847 |
| LogReg Embeddings | 0.6483 | 0.7100 | 0.9045 | 0.7437 |

El DistilBERT mejora el F1-Macro en 3.3 puntos porcentuales sobre el baseline y en 4.7 puntos en la clase neutral, que es la mas dificil por ser minoritaria y semanticamente ambigua.

### Fase 4 — MLOps y Recuperacion Semantica

Pipeline de recuperacion semantica sobre resenas de electronica con FAISS y embeddings multilingues (`paraphrase-multilingual-MiniLM-L12-v2`). El sistema permite consultas en espanol sobre un corpus en ingles sin traduccion previa.

Resultados sobre 7 queries de prueba:

| Metrica | Resultado | Objetivo |
|---------|-----------|----------|
| Precision@5 media | 0.80 | >= 0.60 |
| MRR media | 0.93 | >= 0.60 |
| Latencia media | 22 ms | < 200 ms |

Todos los objetivos superados. La comparacion con TF-IDF mostro que los embeddings MiniLM son superiores en queries donde la semantica importa mas que la coincidencia lexica, y son el unico enfoque viable para busqueda cross-language.

El pipeline MLOps incluye tracking con MLflow, versionado del indice FAISS, politica de actualizacion con tres triggers (antiguedad > 30 dias, drift de calidad, volumen de datos nuevos > 10%) y retencion de las ultimas 3 versiones.

---

## Notas de reproducibilidad

Los archivos `.joblib` del pipeline y el modelo LightGBM fueron serializados con scikit-learn 1.8 y cuML/RAPIDS en GPU. Para cargarlos en CPU sin RAPIDS instalado, el repositorio incluye `demo/routes/migrar_pipeline_cpu.py` con las clases equivalentes en sklearn puro. Este archivo debe estar presente antes de ejecutar `demo/main.py`.

El adaptador LoRA del transformer (`models/distilbert_lora/`) debe descargarse por separado desde el Drive compartido del equipo y colocarse en esa ruta antes de iniciar el servidor.

Los archivos de datos (`data/*.joblib`, `data/*.csv`) no estan incluidos en el repositorio por su tamano. Contactar al equipo para acceso.

---

## Stack tecnologico

| Libreria | Uso |
|----------|-----|
| FastAPI + Uvicorn | API REST y servidor de la demo |
| scikit-learn | Pipeline CPU, TF-IDF, StandardScaler, metricas |
| LightGBM | Modelo baseline de sentimiento |
| sentence-transformers | Embeddings MiniLM para RAG |
| FAISS (faiss-cpu) | Indice vectorial de busqueda exacta |
| transformers + peft | DistilBERT con adaptador LoRA |
| MLflow | Tracking de experimentos y versionado |
| pandas + numpy + scipy | Procesamiento de datos |
| cuML / RAPIDS | Entrenamiento en GPU (solo notebooks, no produccion) |
| FLAML | AutoML benchmark en Fase 2 |
| Optuna | Optimizacion de hiperparametros |
| UMAP-learn | Reduccion de dimensionalidad de embeddings |

---

## Consideraciones eticas

El dataset contiene resenas de usuarios reales de Amazon. No se incluye informacion de identificacion personal. El subconjunto usado fue muestreado aleatoriamente y el criterio de muestreo esta documentado en `data/FUENTE.md`.

El modelo de sentimiento tiene sesgo hacia la clase positiva por el desbalance natural del dataset (74% positivo). Se uso F1-Macro como metrica principal precisamente para penalizar este comportamiento. En produccion, las predicciones de la clase neutral deben interpretarse con cautela dado su menor rendimiento (F1-Neutral baseline: 0.39, transformer: 0.43).

La busqueda semantica devuelve fragmentos de resenas tal como fueron escritos por los usuarios. El sistema no genera ni modifica texto.

---

Proyecto academico — Machine Learning 2026-I — UCB San Pablo