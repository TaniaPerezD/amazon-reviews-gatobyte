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
    df = pd.read_csv(BASE_DIR / "data/umap_coords.csv")
    df = df.where(pd.notna(df), None)
    return df.rename(columns={"UMAP1":"umap1","UMAP2":"umap2"}).to_dict(orient="records")