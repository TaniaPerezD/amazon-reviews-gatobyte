from fastapi import APIRouter
from pathlib import Path

# 1. ESTA LÍNEA ES CRUCIAL (Define el router antes de usarlo)
router = APIRouter(prefix="/api/embeddings", tags=["Embeddings"])

# Definimos BASE_DIR si no lo tenías para que no falle el CSV
BASE_DIR = Path(__file__).resolve().parent.parent.parent

@router.get("/")
async def test_embeddings():
    return {"message": "Ruta de embeddings funcionando"}

@router.get("/umap")
async def get_umap():
    import pandas as pd
    umap_path = BASE_DIR / "data/umap_coords.csv"
    if not umap_path.exists():
        return {"error": "umap_coords.csv no encontrado. Genera el archivo primero."}
    df = pd.read_csv(umap_path)
    df = df.where(pd.notna(df), None)
    return df.rename(columns={"UMAP1":"umap1","UMAP2":"umap2"}).to_dict(orient="records")