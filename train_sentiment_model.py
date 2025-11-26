import pandas as pd
import spacy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model for the spaCy POS tagger\n"
        "(don't worry, this will only happen once)")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

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

def train_model():
    print("Loading dataset...")
    try:
        df = pd.read_csv("Emotion_classify_Data.csv")
    except FileNotFoundError:
        print("Error: 'Emotion_classify_Data.csv' not found.")
        return

    print("Preprocessing data (this may take a while)...")
    # Apply preprocessing
    df['processed_text'] = df['Comment'].apply(preprocess)
    
    X = df['processed_text']
    y = df['Emotion']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training model...")
    # Create a pipeline with TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    model_filename = "sentiment_model.pkl"
    joblib.dump(pipeline, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    train_model()
