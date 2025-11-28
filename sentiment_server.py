
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import os

app = FastAPI()

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

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

def preprocess_text(text):
    """
    Preprocesses text using NLTK: tokenization, stopword removal, lemmatization.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenization & Lowercasing
    tokens = word_tokenize(text.lower())
    
    # Filtering and Lemmatization
    filtered_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word.isalnum() and word not in stop_words
    ]
    
    return ' '.join(filtered_tokens)

@app.post("/analyze")
async def analyze_sentiment(request: SentimentRequest):
    if not model_pipeline:
        # Fallback mock logic if model is not loaded
        sentiment = "positive" if "good" in request.text.lower() else "negative"
        return {"sentiment": sentiment, "score": 0.99, "model": "mock_fallback"}

    try:
        processed_text = preprocess_text(request.text)
        prediction = model_pipeline.predict([processed_text])[0]
        # Get probability if possible (LogisticRegression supports predict_proba)
        try:
            proba = model_pipeline.predict_proba([processed_text]).max()
        except:
            proba = 1.0
            
        return {"sentiment": prediction, "score": float(proba), "model": "custom_nltk_sklearn"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
