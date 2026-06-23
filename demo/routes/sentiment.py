from fastapi import APIRouter

# Definir el router que main.py va a buscar
router = APIRouter(prefix="/api/sentiment", tags=["Sentiment"])

@router.get("/")
async def test_sentiment():
    return {"message": "Ruta de sentimiento funcionando"}