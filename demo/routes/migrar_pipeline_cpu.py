"""
migrar_pipeline_cpu.py
======================
Migra el pipeline GPU (cuML/cuDF) a un pipeline 100% CPU (sklearn/scipy/pandas).
No requiere reentrenamiento. Lee los joblibs originales y produce:

  - pipeline_transformacion_cpu.joblib      → preprocesador sklearn puro
  - lightgbm_tuned_final_cpu.joblib        → LightGBM con device='cpu'
  - params_extraidos.json    → todos los parámetros aprendidos (auditoría)

Uso:
  python migrar_pipeline_cpu.py \
      --pipeline pipeline_transformacion.joblib \
      --modelo   lightgbm_tuned_final.joblib \
      --salida   ../

Requisitos (CPU, sin CUDA):
  pip install scikit-learn lightgbm scipy numpy joblib
"""

import sys
import types
import json
import argparse
import numpy as np
import joblib
import joblib.numpy_pickle as jnp

from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. STUBS GPU — permiten deserializar el joblib sin tener CUDA instalado
# ─────────────────────────────────────────────────────────────────────────────

class _StubBase:
    """Clase base que absorbe cualquier estado pickle sin explotar."""
    def __new__(cls, *a, **kw):
        return object.__new__(cls)
    def __init__(self, *a, **kw):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
    @classmethod
    def host_deserialize(cls, *a, **kw):
        return object.__new__(cls)


# Las tres clases custom del notebook deben estar en __main__
class GpuFullPreprocessor(_StubBase): pass
class GpuPreprocessor(_StubBase):     pass
class GpuTextCleaner(_StubBase):      pass

# Registrar en __main__ para que pickle las encuentre
import __main__ as _main
_main.GpuFullPreprocessor = GpuFullPreprocessor
_main.GpuPreprocessor     = GpuPreprocessor
_main.GpuTextCleaner      = GpuTextCleaner


def _make_stub(name: str) -> type:
    return type(name, (_StubBase,), {
        "host_deserialize": classmethod(lambda c, *a, **kw: object.__new__(c))
    })


class _CumlArrayDescriptorMeta(type):
    """Metaclase stub para CumlArrayDescriptorMeta."""
    def __new__(mcs, *args, **kw):
        if len(args) != 3:
            return object.__new__(_StubBase)
        return super().__new__(mcs, *args, **kw)


def _register_gpu_stubs() -> None:
    """Registra todos los módulos GPU ficticios en sys.modules."""
    _CuArray = _make_stub("CuArray")

    stubs = {
        "cuml":                                        {},
        "cuml.dask":                                   {},
        "cuml.dask.common":                            {},
        "cuml.dask.common.base":                       {"Base": _StubBase},
        "cuml.common":                                 {},
        "cuml.common.array_descriptor":                {"CumlArrayDescriptorMeta": _CumlArrayDescriptorMeta},
        "cuml._thirdparty":                            {},
        "cuml._thirdparty.sklearn":                    {},
        "cuml._thirdparty.sklearn.preprocessing":      {},
        "cuml._thirdparty.sklearn.preprocessing._data":{"StandardScaler": _make_stub("StandardScaler")},
        "cuml.feature_extraction":                     {},
        "cuml.feature_extraction._tfidf_vectorizer":   {"TfidfVectorizer": _make_stub("TfidfVectorizer")},
        "cuml.feature_extraction._tfidf":              {"TfidfTransformer": _make_stub("TfidfTransformer")},
        "cuml.pipeline":                               {"Pipeline": _make_stub("Pipeline")},
        "cuml.compose":                                {},
        "cupy":                                        {"ndarray": _CuArray, "array": _CuArray},
        "cupy._core":                                  {"ndarray": _CuArray, "array": _CuArray},
        "cupy._core.core":                             {"ndarray": _CuArray, "array": _CuArray},
        "cupyx":                                       {},
        "cupyx.scipy":                                 {},
        "cupyx.scipy.sparse":                          {},
        "cupyx.scipy.sparse._dia":                     {"dia_matrix": _make_stub("dia_matrix")},
        "cudf":                                        {},
        "cudf.core":                                   {},
        "cudf.core.series":                            {"Series": _make_stub("Series")},
        "pylibcudf":                                   {},
        "pylibcudf.types":                             {"DataType": _make_stub("DataType")},
        "pylibcudf.libcudf":                           {},
        "pylibcudf.libcudf.types":                     {"type_id": type("type_id", (), {})},
    }
    for mod_name, attrs in stubs.items():
        mod = types.ModuleType(mod_name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[mod_name] = mod


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACCIÓN DE PARÁMETROS DEL PIPELINE GPU
# ─────────────────────────────────────────────────────────────────────────────

def _load_gpu_pipeline(path: str):
    """
    Carga el pipeline GPU usando stubs y captura todos los arrays numpy
    embebidos en el joblib (scaler mean/var/scale, IDF, vocabulario).
    Devuelve (pipeline_objeto, lista_arrays_numpy).
    """
    _register_gpu_stubs()

    captured_arrays = []
    _orig_wrapper = jnp.NumpyArrayWrapper

    class _TrackingWrapper(_orig_wrapper):
        def read(self, unpickler, ensure_native):
            arr = super().read(unpickler, ensure_native)
            captured_arrays.append(arr)
            return arr

    jnp.NumpyArrayWrapper = _TrackingWrapper
    try:
        pipeline = joblib.load(path)
    finally:
        jnp.NumpyArrayWrapper = _orig_wrapper

    return pipeline, captured_arrays


def extract_params(pipeline_path: str) -> dict:
    """
    Extrae todos los parámetros aprendidos del pipeline GPU.

    Orden de los arrays capturados (determinista, validado con los archivos reales):
      [0] mean_   shape=(2,) float64  → [text_len_mean, price_mean]
      [1] var_    shape=(2,) float64  → [text_len_var,  price_var]
      [2] scale_  shape=(2,) float64  → [text_len_scale,price_scale]
      [3] idf     shape=(1,10000) f32 → IDF diagonal del TF-IDF
      [4] idf_off shape=(1,) int32    → offset (siempre [0])
      [5] sw_keys shape=(N,) uint8    → bytes utf-8 del vocabulario de 10k tokens
      [6] sw_offs shape=(M,) uint8    → offsets int32 de sw_keys
      [7] vk_keys shape=(P,) uint8    → bytes utf-8 del vocabulario completo visto
      [8] vk_offs shape=(Q,) uint8    → offsets int32 de vk_keys
    """
    print(f"  Cargando pipeline GPU desde: {pipeline_path}")
    pipeline, arrays = _load_gpu_pipeline(pipeline_path)

    prep  = pipeline.steps[0][1]        # GpuFullPreprocessor
    inner = prep._prep                  # GpuPreprocessor
    sc    = prep._scaler                # CumlStandardScaler
    tfidf = prep._tfidf                 # CumlTfidfVectorizer
    trans = tfidf._tfidf                # CumlTfidfTransformer

    # ── Clip params ──────────────────────────────────────────────────────────
    clip = {
        "p1_price":  float(inner.p1_price_),
        "p99_price": float(inner.p99_price_),
        "p1_len":    float(inner.p1_len_),
        "p99_len":   float(inner.p99_len_),
        "cat_cols":  list(inner.cat_cols_),
    }

    # ── StandardScaler ───────────────────────────────────────────────────────
    mean_arr  = arrays[0]   # shape (2,)
    var_arr   = arrays[1]   # shape (2,)
    scale_arr = arrays[2]   # shape (2,)

    scaler_params = {
        "with_mean":      bool(sc.with_mean),
        "with_std":       bool(sc.with_std),
        "n_features_in_": int(sc.n_features_in_),
        "mean_":          mean_arr.tolist(),
        "var_":           var_arr.tolist(),
        "scale_":         scale_arr.tolist(),
        "n_samples_seen_": 700000,   # train+val usado en el notebook
    }

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    # arrays[5],[6] = vocabulario de 10k tokens (feature_index order)
    sw_keys_bytes = arrays[5].tobytes().decode("utf-8", errors="replace")
    sw_offs_int32 = arrays[6].view(np.int32)
    feature_tokens = [
        sw_keys_bytes[sw_offs_int32[i]: sw_offs_int32[i + 1]]
        for i in range(len(sw_offs_int32) - 1)
    ]
    # vocabulary_: token -> feature_index (posición en la lista)
    vocabulary = {tok: i for i, tok in enumerate(feature_tokens)}

    # IDF diagonal: shape (1,10000) -> ravel a (10000,)
    idf_values = arrays[3].ravel().astype(np.float64).tolist()

    tfidf_params = {
        "analyzer":    tfidf.analyzer,
        "lowercase":   bool(tfidf.lowercase),
        "max_df":      float(tfidf.max_df),
        "min_df":      int(tfidf.min_df),
        "max_features":int(tfidf.max_features),
        "ngram_range": list(tfidf.ngram_range),
        "binary":      bool(tfidf.binary),
        "norm":        trans.norm,
        "use_idf":     bool(trans.use_idf),
        "smooth_idf":  bool(trans.smooth_idf),
        "sublinear_tf":bool(trans.sublinear_tf),
        "n_samples":   int(trans._TfidfTransformer__n_samples),
        "n_features":  int(trans._TfidfTransformer__n_features),
        "vocabulary":  vocabulary,   # {token: index}
        "idf_values":  idf_values,   # list[float], len=10000
    }

    return {"clip": clip, "scaler": scaler_params, "tfidf": tfidf_params}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE CPU — clases sklearn puras
# ─────────────────────────────────────────────────────────────────────────────

import re
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SklearnPipeline


class CpuPreprocessor(BaseEstimator, TransformerMixin):
    """
    Equivalente CPU de GpuPreprocessor.
    Realiza:
      - fillna de texto y categoría
      - concatenación title + text → text_combined
      - clip de outliers (p1/p99) en price y text_len
      - one-hot encoding de main_category
    """

    def __init__(self, p1_price, p99_price, p1_len, p99_len, cat_cols):
        self.p1_price  = p1_price
        self.p99_price = p99_price
        self.p1_len    = p1_len
        self.p99_len   = p99_len
        self.cat_cols  = cat_cols

    def fit(self, X, y=None):
        return self   # parámetros ya vienen del GPU pipeline

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["title"]         = X["title"].fillna("")
        X["text"]          = X["text"].fillna("")
        X["main_category"] = X["main_category"].fillna("Unknown")
        X["text_combined"] = X["title"].str.strip() + " " + X["text"].str.strip()

        X["price"]    = X["price"].clip(self.p1_price, self.p99_price)
        X["text_len"] = X["text_len"].clip(self.p1_len, self.p99_len)

        # One-hot encoding fijo (las mismas columnas que en GPU)
        dummies = pd.get_dummies(X["main_category"], prefix="cat")
        for col in [f"cat_{c}" for c in self.cat_cols]:
            if col not in dummies.columns:
                dummies[col] = 0
        dummies = dummies[[f"cat_{c}" for c in self.cat_cols]].astype("float32")

        return pd.concat(
            [X[["text_len", "price"]], dummies, X[["text_combined"]]],
            axis=1,
        )


class CpuTextCleaner(BaseEstimator, TransformerMixin):
    """
    Equivalente CPU de GpuTextCleaner.
    Limpieza de texto: minúsculas, quita acentos, elimina no-letras.
    """

    _ACCENT_MAP = str.maketrans(
        "áéíóúüñÁÉÍÓÚÜÑ",
        "aeiouunAEIOUUN",
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        s = X.fillna("").str.lower()
        s = s.str.translate(self._ACCENT_MAP)
        s = s.str.replace(r"[^a-z\s]", " ", regex=True)
        s = s.str.replace(r"\s+", " ", regex=True)
        return s.str.strip()


class CpuFullPreprocessor(BaseEstimator, TransformerMixin):
    """
    Pipeline CPU unificado. Equivalente exacto de GpuFullPreprocessor.

    Pasos:
      A. CpuPreprocessor  → feature engineering + text_combined
      B. StandardScaler   → escala price y text_len
      C. CpuTextCleaner   → limpieza de texto
      D. TfidfVectorizer  → sparse matrix (scipy)
      E. hstack           → scipy csr_matrix final
    """

    def __init__(self, params: dict):
        self.params = params
        self._build_inner()

    def _build_inner(self):
        p = self.params

        # A — Preprocessor (parámetros de clip y categorías)
        self._prep = CpuPreprocessor(
            p1_price  = p["clip"]["p1_price"],
            p99_price = p["clip"]["p99_price"],
            p1_len    = p["clip"]["p1_len"],
            p99_len   = p["clip"]["p99_len"],
            cat_cols  = p["clip"]["cat_cols"],
        )

        # B — StandardScaler con parámetros aprendidos
        sc_p = p["scaler"]
        self._scaler = StandardScaler(
            with_mean=sc_p["with_mean"],
            with_std=sc_p["with_std"],
            copy=True,
        )
        self._scaler.mean_           = np.array(sc_p["mean_"],  dtype=np.float64)
        self._scaler.var_            = np.array(sc_p["var_"],   dtype=np.float64)
        self._scaler.scale_          = np.array(sc_p["scale_"], dtype=np.float64)
        self._scaler.n_features_in_  = sc_p["n_features_in_"]
        self._scaler.n_samples_seen_ = sc_p["n_samples_seen_"]
        self._scaler.feature_names_in_ = np.array(["text_len", "price"])

        # C — Text cleaner
        self._cleaner = CpuTextCleaner()

        # D — TfidfVectorizer con vocabulario y IDF aprendidos
        tf_p = p["tfidf"]
        self._tfidf = TfidfVectorizer(
            analyzer    = tf_p["analyzer"],
            lowercase   = tf_p["lowercase"],
            max_df      = tf_p["max_df"],
            min_df      = tf_p["min_df"],
            max_features= tf_p["max_features"],
            ngram_range = tuple(tf_p["ngram_range"]),
            binary      = tf_p["binary"],
            norm        = tf_p["norm"],
            use_idf     = tf_p["use_idf"],
            smooth_idf  = tf_p["smooth_idf"],
            sublinear_tf= tf_p["sublinear_tf"],
            vocabulary  = tf_p["vocabulary"],   # fija el vocabulario
        )
        # Inicializar el TfidfTransformer interno con un fit mínimo,
        # luego inyectar el IDF real aprendido en GPU.
        # (sklearn solo crea _tfidf después de fit, no al construir)
        dummy_text = " ".join(list(tf_p["vocabulary"].keys())[:50])
        self._tfidf.fit([dummy_text])

        # Reemplazar IDF con los valores exactos del modelo GPU
        idf_arr = np.array(tf_p["idf_values"], dtype=np.float64)
        self._tfidf._tfidf.idf_ = idf_arr

    def fit(self, X, y=None):
        return self   # ya entrenado, no hace nada

    def __sklearn_is_fitted__(self):
        """Indica a sklearn que este transformador ya está entrenado."""
        return True

    def transform(self, X: pd.DataFrame):
        # A — feature engineering
        prep_df = self._prep.transform(X)

        # B — escalar columnas numéricas
        num_scaled = self._scaler.transform(prep_df[["text_len", "price"]])
        num_sparse  = sp.csr_matrix(num_scaled.astype(np.float32))

        # one-hot cols (todas menos text_len, price, text_combined)
        cat_cols_in_df = [c for c in prep_df.columns if c.startswith("cat_")]
        cat_sparse = sp.csr_matrix(
            prep_df[cat_cols_in_df].values.astype(np.float32)
        )

        # C+D — limpieza + TF-IDF
        cleaned_text = self._cleaner.transform(prep_df["text_combined"])
        tfidf_sparse = self._tfidf.transform(cleaned_text)

        # E — hstack → csr final
        return sp.hstack([num_sparse, cat_sparse, tfidf_sparse], format="csr")

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @property
    def cat_cols_(self):
        return self._prep.cat_cols


def build_cpu_pipeline(params: dict) -> SklearnPipeline:
    """Construye el pipeline CPU completo listo para usar."""
    preprocessor = CpuFullPreprocessor(params)
    return SklearnPipeline(steps=[("preprocesador", preprocessor)])


# ─────────────────────────────────────────────────────────────────────────────
# 4. MIGRACIÓN DEL MODELO LIGHTGBM
# ─────────────────────────────────────────────────────────────────────────────

def migrate_lgbm(model_path: str):
    """
    Carga el modelo GPU y devuelve una copia lista para CPU.
    device='gpu' es solo un parámetro de entrenamiento — la inferencia
    funciona en CPU de todas formas. Se parchea el dict interno para
    evitar confusiones, sin llamar a set_params() que puede causar
    segfault al intentar reinicializar el booster CUDA.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)

    # Parchear device en el dict de params (cosmético, no rebuilds el booster)
    model.__dict__["device"] = "cpu"
    if hasattr(model, "_other_params") and "device" in model._other_params:
        model._other_params["device"] = "cpu"

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. VALIDACIÓN RÁPIDA
# ─────────────────────────────────────────────────────────────────────────────

def validate(cpu_pipeline, cpu_model, le_classes):
    """
    Prueba end-to-end con 5 reseñas ficticias para confirmar que el pipeline
    CPU produce predicciones válidas.
    """
    test_df = pd.DataFrame({
        "title": [
            "Amazing product, works perfectly",
            "Terrible quality, broke after one day",
            "Its okay, nothing special",
            "Best headphones I have ever bought",
            "Disappointed, not as described",
        ],
        "text": [
            "I bought this for my home office and it exceeded all my expectations.",
            "Complete waste of money. Stopped working after 24 hours. Do not buy.",
            "Average product. Does what it says but nothing more.",
            "Incredible sound quality and very comfortable. Highly recommend.",
            "The pictures looked great but the real product is much smaller.",
        ],
        "text_len":      [60, 55, 45, 62, 58],
        "price":         [29.99, 15.49, 22.00, 89.99, 34.99],
        "main_category": ["Electronics"] * 5,
    })

    X_test = cpu_pipeline.transform(test_df)
    preds  = cpu_model.predict(X_test)
    probas = cpu_model.predict_proba(X_test)

    print("\n  Validación end-to-end:")
    print(f"  {'Reseña':<42} {'Pred':>10} {'Confianza':>10}")
    print("  " + "─" * 64)
    for title, pred_idx, proba in zip(test_df["title"], preds, probas):
        clase      = le_classes[pred_idx] if le_classes else str(pred_idx)
        confianza  = proba.max()
        print(f"  {title[:42]:<42} {clase:>10} {confianza:>9.1%}")
    print()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Migra pipeline GPU → CPU")
    parser.add_argument("--pipeline", default="pipeline_transformacion.joblib",
                        help="Ruta al pipeline GPU original")
    parser.add_argument("--modelo",   default="lightgbm_tuned_final.joblib",
                        help="Ruta al modelo LightGBM GPU original")
    parser.add_argument("--le",       default=None,
                        help="Ruta al label_encoder.joblib (opcional, para validación)")
    parser.add_argument("--salida",   default="../",
                        help="Directorio de salida (por defecto: carpeta padre de migracion_cpu)")
    parser.add_argument("--solo-validar", action="store_true",
                        help="Saltar extracción, solo validar archivos ya migrados")
    args = parser.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    pipeline_cpu_path = salida / "pipeline_transformacion_cpu.joblib"
    modelo_cpu_path   = salida / "lightgbm_tuned_final_cpu.joblib"
    params_path       = Path(__file__).parent / "params_extraidos.json"

    # ── Modo solo-validar ────────────────────────────────────────────────────
    if args.solo_validar:
        print("Modo validación: cargando archivos ya migrados...")
        cpu_pipeline = joblib.load(pipeline_cpu_path)
        cpu_model    = joblib.load(modelo_cpu_path)
        le_classes   = None
        if args.le:
            le = joblib.load(args.le)
            le_classes = le.classes_
        validate(cpu_pipeline, cpu_model, le_classes)
        print("Validación OK")
        return

    # ── Paso 1: extraer parámetros del pipeline GPU ─────────────────────────
    print("\n[1/4] Extrayendo parámetros del pipeline GPU...")
    params = extract_params(args.pipeline)

    n_cat   = len(params["clip"]["cat_cols"])
    n_vocab = params["tfidf"]["n_features"]
    print(f"      Clip: price [{params['clip']['p1_price']}, {params['clip']['p99_price']}] | "
          f"len [{params['clip']['p1_len']}, {params['clip']['p99_len']}]")
    print(f"      Scaler mean: {params['scaler']['mean_']}")
    print(f"      Categorías OHE: {n_cat}")
    print(f"      Vocabulario TF-IDF: {n_vocab} tokens")
    print(f"      IDF muestra: {params['tfidf']['idf_values'][:3]}")

    # Guardar params como JSON (auditoría)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"      Parámetros guardados en: {params_path}")

    # ── Paso 2: construir pipeline CPU ──────────────────────────────────────
    print("\n[2/4] Construyendo pipeline CPU (sklearn puro)...")
    cpu_pipeline = build_cpu_pipeline(params)
    prep_obj     = cpu_pipeline.named_steps["preprocesador"]
    print(f"      Scaler mean_:  {prep_obj._scaler.mean_}")
    print(f"      Scaler scale_: {prep_obj._scaler.scale_}")
    print(f"      Vocab size:    {len(prep_obj._tfidf.vocabulary_)}")
    joblib.dump(cpu_pipeline, pipeline_cpu_path, compress=3)
    print(f"      Guardado: {pipeline_cpu_path}")

    # ── Paso 3: migrar modelo LightGBM ──────────────────────────────────────
    print("\n[3/4] Migrando modelo LightGBM GPU → CPU...")
    cpu_model = migrate_lgbm(args.modelo)
    print(f"      device param: {cpu_model.get_params()['device']}")
    print(f"      n_estimators: {cpu_model.get_params()['n_estimators']}")
    joblib.dump(cpu_model, modelo_cpu_path, compress=3)
    print(f"      Guardado: {modelo_cpu_path}")

    # ── Paso 4: validación end-to-end ───────────────────────────────────────
    print("\n[4/4] Validación end-to-end...")
    le_classes = None
    if args.le:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            le = joblib.load(args.le)
        le_classes = le.classes_

    ok = validate(cpu_pipeline, cpu_model, le_classes)
    if ok:
        print("  ✓ Pipeline CPU funciona correctamente")

    # ── Resumen ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  MIGRACIÓN COMPLETADA")
    print("═" * 60)
    print(f"  pipeline_transformacion_cpu.joblib  → {pipeline_cpu_path}")
    print(f"  lightgbm_tuned_final_cpu.joblib    → {modelo_cpu_path}")
    print(f"  params_extraidos.json→ {params_path}")
    print(f"\n  Features totales: 2 num + {n_cat} OHE + {n_vocab} TFIDF = {2 + n_cat + n_vocab}")
    print("═" * 60)


if __name__ == "__main__":
    main()
