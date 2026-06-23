# Nota de procedencia del dataset — GATOBYTE

## Archivo principal
**Nombre:** `sample_ml.parquet`  
**Ubicación en Colab:** `drive/My Drive/Machine Learning/proyecto/sample_ml.parquet`

## Origen
- **Dataset base:** Amazon Reviews 2023  
- **Fuente académica:** McAuley Lab, UC San Diego  
- **Publicación:** "Bridging Language and Items for Retrieval and Recommendation" (Hou et al., 2024, arXiv:2403.03952)  
- **Repositorio oficial:** https://amazon-reviews-2023.github.io/  
- **Hugging Face:** https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

## Descripción de la muestra
- **Registros:** 1 000 000 (subconjunto estratificado de Electronics)
- **Columnas totales:** 20
- **Período temporal:** mayo 1996 – septiembre 2023
- **Categorías incluidas:** All Electronics, Computers, Cell Phones & Accessories, Camera & Photo

## Columnas usadas por el módulo RAG
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `text` | object | Texto completo de la reseña (columna a indexar) |
| `title` | object | Título de la reseña |
| `rating` | float64 | Calificación 1-5 estrellas |
| `helpful_vote` | int64 | Votos de utilidad recibidos |
| `parent_asin` | object | Identificador del producto padre |
| `sentiment` | object | Sentimiento calculado en Fase 2 (positive/neutral/negative) |
| `main_category` | object | Categoría principal del producto |

## Preprocesamiento previo (aplicado en Fases 2 y 3)
El parquet ya incluye las columnas `sentiment` e `is_satisfied` calculadas
en el pipeline de Fase 2, por lo que el módulo RAG las utiliza directamente
como metadatos sin riesgo de data leakage (no se usan como predictores, 
solo como filtros de metadatos en los resultados de búsqueda).

## Riesgos documentados
- **Sesgo de selección:** muestra puede no ser representativa de todas las subcategorías
- `main_category` presenta 16 834 registros nulos (1.68%) — tratados como string vacío en el RAG
- Evolución temporal del lenguaje (1996-2023) puede afectar coherencia semántica del índice
