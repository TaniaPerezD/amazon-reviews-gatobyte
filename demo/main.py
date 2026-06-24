"""
=============================================================
 GATOBYTE — Mini Proyecto MLOps + RAG
 /demo/main.py
=============================================================
"""
import sys, os, json, pickle, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── REGISTRO DE CLASES PARA JOBLIB ──────────────────
# Debe ir ANTES de importar los routers que cargan joblib.
# El pipeline fue serializado con __main__ como módulo,
# asi que las clases deben estar en sys.modules['__main__'].
import re, unicodedata
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CpuFullPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs): pass
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class CpuPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs): pass
    def fit(self, X, y=None): return self
    def transform(self, X): return X
    
class CpuTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs): pass
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# Registrar en __main__ (que es este mismo archivo cuando uvicorn lo carga)
import __main__
__main__.CpuFullPreprocessor = CpuFullPreprocessor
__main__.CpuPreprocessor     = CpuPreprocessor
__main__.CpuTextCleaner      = CpuTextCleaner
# ────────────────────────────────────────────────────
import sys
import os
from pathlib import Path

# Asegurar que los imports relativos funcionen desde la raíz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# 1. IMPORTAMOS TUS ROUTERS
# DESPUES
from demo.routes import sentiment, embeddings

try:
    from demo.routes import rag
    app.include_router(rag.router)
    print("Router RAG cargado.")
except Exception as e:
    print(f"Router RAG no disponible: {e}")

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"

# ─────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────
app = FastAPI(title="GATOBYTE Review Search API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─────────────────────────────────────────────────────────
# 2. INCLUIMOS LOS ROUTERS (¡Lo que tú propusiste!)
# ─────────────────────────────────────────────────────────
app.include_router(rag.router)
app.include_router(sentiment.router)
app.include_router(embeddings.router)


# ─────────────────────────────────────────────────────────
# Rutas Raíz
# ─────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    uvicorn.run("demo.main:app", host="0.0.0.0", port=7860, reload=True)