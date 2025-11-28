import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

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

def train_model():
    print("Loading dataset...")
    try:
        df = pd.read_csv("Emotion_classify_Data.csv")
    except FileNotFoundError:
        print("Error: 'Emotion_classify_Data.csv' not found.")
        return

    print("Preprocessing data (this may take a while)...")
    # Apply preprocessing
    df['processed_text'] = df['Comment'].apply(preprocess_text)
    
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
