"""
=============================================================
 GATOBYTE — Mini Proyecto MLOps + RAG
 /demo/main.py
=============================================================
"""

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
from demo.routes import rag, sentiment, embeddings

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