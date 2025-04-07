# Sentiment Analysis and Image Classification API

This project is a FastAPI-based application that provides two core machine learning functionalities:

1. **Sentiment Analysis** on user-provided text using a DistilBERT model fine-tuned on the SST-2 dataset.
2. **Image Classification** on uploaded image files using a Vision Transformer (ViT) model.

The application is structured to be simple, modular, and easy to extend. It uses pre-trained models from the Hugging Face Transformers library.

## Features

- REST API built using FastAPI
- Sentiment analysis of English sentences
- Image classification using the ViT transformer model
- Easy testing through the built-in Swagger UI provided by FastAPI

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MostafaJammoul/C2.2_Asst.git
```

2. Create a virtual environment and activate it (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## File Overview

### requirements.txt
Contains all necessary dependencies for running the application:
- `transformers`: Hugging Face library for pre-trained models
- `torch`: PyTorch backend
- `fastapi`: API framework
- `uvicorn`: ASGI server
- `pillow`: For image processing
- `python-multipart`: For handling image uploads
- `pydantic`: For data validation

### imagefeature.py
Defines an API router for handling image classification. It includes:
- `ImageClassifier` class to initialize and handle predictions from the ViT model
- A route `/image` which accepts image uploads and returns the predicted label

### SentimentAnalysis_DBERT.py
Defines an API router for text sentiment analysis. It includes:
- A `PredictionRequest` data model to validate input text
- A route `/sentiment` that processes input text and returns the sentiment prediction

### main.py
The main entry point of the application:
- Combines the image and sentiment routes
- Defines the root endpoint `/` for welcoming the user, and providing the homepage
- Runs the application on `localhost:8000`

## Running the Application

To start the application, run the following command (Uvicorn is automatically run from the main.py script):

```bash
python main.py
```

Once the server is running, navigate to:

- `http://127.0.0.1:8000` - Root endpoint
- `http://127.0.0.1:8000/docs` -  Built-in Swagger UI for testing the API endpoints

## API Endpoints

### POST /sentiment
- **Description**: Accepts a JSON object with a `text` field. Returns the predicted sentiment.
- **Input Example**:
```json
{
  "text": "I love this product!"
}
```
- **Output Example**:
```json
{
  "From your input, your sentiment is most probably": [
    {
      "label": "POSITIVE",
      "score": 0.9998
    }
  ]
}
```

### POST /image
- **Description**: Accepts an image file. Returns the predicted class label.
- **Form Field**: `file`
- **Output Example**:
```json
{
  "predicted_label": "Labrador retriever"
}
```

## Notes
- Only `.jpg`, `.png`, or other standard image formats are supported.
- The application uses CPU by default. For GPU acceleration, additional setup with CUDA is needed.

## Future Improvements
- Nothing. This is just an assignment!

## License
This project is only for the assignment and related purposes.

## Notes
I collaborated with Mohammad Chmaitilly, where we both pushed our models to each others' repositories on GitHub. The image classification model is his while the sentiment analysis model is mine. After pushing, we collaborated on creating a main.py script which utilizes routers to manage each model in its own file (IEP) which are then integrated into one EEP.

## Versioning
I utilized GitHub tags by running:

```bash
git tag -a <version_number> -m <display_message_for_tag>
```

This command adds a tag to the latest commit (at the HEAD), and saves the current timeline for easy navigation between versions.

Then, we run:

```bash
git push origin --tags
```

And this pushes our tag (with the corresponding commit) to GitHub. 
On GitHub, we can click on "Tags," then we see a list of version numbers. Clicking on any of them, we can download the sourcecode zip file which includes all files at the time of that tag push. This allows for consistent and easy versioning, and even rollback in case anything unexpected occurs.
