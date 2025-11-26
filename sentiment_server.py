
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import joblib
import os

app = FastAPI()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model for the spaCy POS tagger")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load the trained model
MODEL_PATH = "sentiment_model.pkl"
model_pipeline = None

if os.path.exists(MODEL_PATH):
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Please run train_sentiment_model.py first.")

class SentimentRequest(BaseModel):
    text: str

def preprocess(text):
    """
    Preprocesses text by removing stop words, punctuation, and lemmatizing.
    """
    doc = nlp(text.lower())
    filtered_tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(filtered_tokens)

@app.post("/analyze")
async def analyze_sentiment(request: SentimentRequest):
    if not model_pipeline:
        # Fallback mock logic if model is not loaded
        sentiment = "positive" if "good" in request.text.lower() else "negative"
        return {"sentiment": sentiment, "score": 0.99, "model": "mock_fallback"}

    try:
        processed_text = preprocess(request.text)
        prediction = model_pipeline.predict([processed_text])[0]
        # Get probability if possible (LogisticRegression supports predict_proba)
        try:
            proba = model_pipeline.predict_proba([processed_text]).max()
        except:
            proba = 1.0
            
        return {"sentiment": prediction, "score": float(proba), "model": "custom_spacy_sklearn"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
