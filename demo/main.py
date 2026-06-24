import sys, os, json, pickle, time, datetime
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # evita segfault torch+lightgbm en macOS
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")      # no descargar nada de HuggingFace
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR   = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"

# ── APP ──────────────────────────────────────────────
app = FastAPI(title="GATOBYTE Review Search API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── ROUTERS ──────────────────────────────────────────
from demo.routes import sentiment, embeddings, metrics
app.include_router(sentiment.router)
app.include_router(embeddings.router)
app.include_router(metrics.router)

try:
    from demo.routes import rag
    app.include_router(rag.router)
    print("Router RAG cargado.")
except Exception as e:
    print(f"Router RAG no disponible: {e}")

# ── RUTAS BASE ───────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("demo.main:app", host="0.0.0.0", port=7862, reload=False)