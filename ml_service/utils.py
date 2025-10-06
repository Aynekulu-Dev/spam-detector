import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import re
import os

def clean_text(text):
    """Basic text cleaning"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_model():
    """Train a simple spam classification model"""
    # Sample spam/ham dataset
    data = {
        'text': [
            'Win free money now! Click here!',
            'Hello, how are you doing today?',
            'Congratulations! You won a $1000 prize!',
            'Lets meet for coffee tomorrow',
            'URGENT: Your account will be suspended',
            'Hi mom, what time are we having dinner?',
            'Free lottery! Claim your prize now!',
            'Meeting rescheduled to 3 PM tomorrow',
            'You are selected for a free gift card',
            'Reminder: Dentist appointment at 2 PM',
            'Limited time offer! Buy now!',
            'Hey, are we still on for lunch?',
            'You have won a free iPhone!',
            'Can you send me the report by EOD?',
            'Exclusive deal just for you!',
            'What is the status of the project?',
            'Get rich quick with this method!',
            'See you at the meeting tomorrow',
            'Act now before its too late!',
            'Thanks for your help with this'
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=spam, 0=ham
    }
    
    df = pd.DataFrame(data)
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Create features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, 'models/spam_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    
    # Calculate accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Training Accuracy: {train_accuracy:.2f}")
    print(f"📊 Test Accuracy: {test_accuracy:.2f}")
    
    return model, vectorizer

# Train model if running directly
if __name__ == "__main__":
    train_model()