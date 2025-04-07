# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

class PredictionRequest(BaseModel):
    text: str

app = FastAPI()

model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.post("/predict")
async def predict(request: PredictionRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    result = model(text)
    return {"From your input, your sentiment is most probably": result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
