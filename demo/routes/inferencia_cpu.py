"""
inferencia_cpu.py
=================
Ejemplo de uso del pipeline CPU migrado para predicción en producción.
No requiere GPU ni RAPIDS instalados.

Uso:
    python inferencia_cpu.py \
        --pipeline pipeline_transformacion_cpu.joblib \
        --modelo   lightgbm_tuned_final_cpu.joblib \
        --le       label_encoder.joblib   # opcional
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import joblib


def predecir(pipeline_path: str, modelo_path: str, le_path: str = None):
    """Carga los modelos CPU y realiza predicciones de ejemplo."""

    print("Cargando artefactos CPU...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = joblib.load(pipeline_path)
        modelo   = joblib.load(modelo_path)
        le       = joblib.load(le_path) if le_path else None

    classes = le.classes_ if le else ["negative", "neutral", "positive"]
    print(f"Clases: {classes}")
    print(f"Features esperadas: {modelo.n_features_in_}")

    # ── Datos de prueba ──────────────────────────────────────────────────────
    reseñas = pd.DataFrame({
        "title": [
            "Amazing product, works perfectly",
            "Terrible quality, broke after one day",
            "Its okay, nothing special",
            "Best headphones I have ever bought",
            "Disappointed, not as described",
            "Solid build, great value for the price",
            "Would not recommend to anyone",
        ],
        "text": [
            "I bought this for my home office and it exceeded all my expectations. Very fast.",
            "Complete waste of money. Stopped working after 24 hours. Do not buy this garbage.",
            "Average product. Does what it says but nothing more. Packaging was damaged.",
            "Incredible sound quality and very comfortable. Worth every penny.",
            "The pictures looked great but the real product is much smaller than expected.",
            "Does exactly what it promises. Durable and well made. Fast shipping too.",
            "Broke after a week. Customer service was unhelpful. Total disappointment.",
        ],
        "text_len":      [70, 75, 55, 60, 65, 62, 58],
        "price":         [29.99, 15.49, 22.00, 89.99, 34.99, 45.00, 18.99],
        "main_category": ["Electronics", "Electronics", "Computers",
                           "Electronics", "Electronics", "Electronics", "Appliances"],
    })

    # ── Preprocesar ──────────────────────────────────────────────────────────
    print("\nPreprocesando...")
    X = pipeline.transform(reseñas)
    print(f"Matriz transformada: {X.shape} (sparse: {X.nnz} valores no-cero)")

    # ── Predecir ─────────────────────────────────────────────────────────────
    preds  = modelo.predict(X)
    probas = modelo.predict_proba(X)

    # ── Mostrar resultados ───────────────────────────────────────────────────
    print(f"\n{'Reseña (título)':<44} {'Predicción':>12} {'Confianza':>10}  Probabilidades")
    print("─" * 100)
    for i, (_, row) in enumerate(reseñas.iterrows()):
        clase     = classes[preds[i]]
        confianza = probas[i].max()
        proba_str = " | ".join(
            f"{classes[j]}={probas[i][j]:.2f}" for j in range(len(classes))
        )
        print(f"  {row['title'][:42]:<42} {clase:>12} {confianza:>9.1%}  {proba_str}")

    return preds, probas, classes


def predecir_una_resena(
    pipeline,
    modelo,
    title: str,
    text: str,
    price: float,
    main_category: str,
    classes: list,
) -> dict:
    """
    Función lista para producción: recibe una reseña y devuelve un dict
    con la predicción y las probabilidades.

    Ejemplo:
        resultado = predecir_una_resena(
            pipeline, modelo,
            title="Great product",
            text="Works as expected. Very happy with the purchase.",
            price=29.99,
            main_category="Electronics",
            classes=["negative", "neutral", "positive"],
        )
        # {"sentiment": "positive", "confidence": 0.94,
        #  "proba": {"negative": 0.02, "neutral": 0.04, "positive": 0.94}}
    """
    df = pd.DataFrame([{
        "title":         title,
        "text":          text,
        "text_len":      len(text),
        "price":         price,
        "main_category": main_category,
    }])

    X      = pipeline.transform(df)
    pred   = modelo.predict(X)[0]
    proba  = modelo.predict_proba(X)[0]

    return {
        "sentiment":  classes[pred],
        "confidence": float(proba.max()),
        "proba":      {cls: float(p) for cls, p in zip(classes, proba)},
    }


def main():
    parser = argparse.ArgumentParser(description="Inferencia CPU con pipeline migrado")
    parser.add_argument("--pipeline", default="pipeline_transformacion_cpu.joblib")
    parser.add_argument("--modelo",   default="lightgbm_tuned_final_cpu.joblib")
    parser.add_argument("--le",       default=None,
                        help="Ruta al label_encoder.joblib (opcional)")
    args = parser.parse_args()

    predecir(args.pipeline, args.modelo, args.le)


if __name__ == "__main__":
    main()
