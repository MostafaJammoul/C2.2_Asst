from fastapi import FastAPI, File, UploadFile
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch
import io

# Class for image classification
class ImageClassifier:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.labels = self.model.config.id2label

    def predict(self, image: Image.Image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
        return self.labels[predicted_class]

# Initialize FastAPI app and classifier
app = FastAPI()
classifier = ImageClassifier()

# Define the prediction endpoint
@app.post("/image")
async def image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    label = classifier.predict(image)
    return {"predicted_label": label}

