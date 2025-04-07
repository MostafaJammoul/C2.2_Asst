# main.py
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from transformers import pipeline

class PredictionRequest(BaseModel):
    text: str

router = APIRouter()
model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@router.post("/sentiment")
async def predict(request: PredictionRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    result = model(text)
    return {"From your input, your sentiment is most probably": result}