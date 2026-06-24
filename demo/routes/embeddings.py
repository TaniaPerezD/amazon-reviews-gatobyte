from fastapi import APIRouter

# Definir el router de embeddings
router = APIRouter(prefix="/api/embeddings", tags=["Embeddings"])

@router.get("/")
async def test_embeddings():
    return {"message": "Ruta de embeddings funcionando"}