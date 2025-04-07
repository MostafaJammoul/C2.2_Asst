from fastapi import FastAPI
from SentimentAnalysis_DBERT import router as sent_router
from imagefeature import router as im_router

app = FastAPI()
app.include_router(im_router)
app.include_router(sent_router)

@app.get("/")
def root_dir():
    return {"message": "Welcome to the sentiment analysis & image classification API"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)