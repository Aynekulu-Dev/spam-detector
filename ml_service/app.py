from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sys

app = FastAPI(title="Spam Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and vectorizer
model = None
vectorizer = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_spam: bool

def clean_text(text):
    """Basic text cleaning"""
    import re
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.on_event("startup")
async def load_model():
    """Load the ML model when the application starts"""
    global model, vectorizer
    try:
        # Try to load pre-trained model
        model_path = 'models/spam_model.pkl'
        vectorizer_path = 'models/vectorizer.pkl'
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            print("✅ ML model loaded successfully!")
        else:
            print("⚠️ Model files not found. Using fallback logic.")
            # Create a simple fallback model
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            
            # Minimal training data
            texts = [
                'win free money now', 'hello how are you',
                'congratulations you won prize', 'meet for coffee tomorrow'
            ]
            labels = [1, 0, 1, 0]  # 1=spam, 0=ham
            
            vectorizer = TfidfVectorizer(max_features=100)
            X = vectorizer.fit_transform(texts)
            
            model = LogisticRegression()
            model.fit(X, labels)
            
            print("✅ Fallback model created successfully!")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.get("/")
async def root():
    return {
        "message": "Spam Detection API", 
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "service": "spam-detection"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if model is None or vectorizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Clean and preprocess text
        cleaned_text = clean_text(request.text)
        
        # Transform text to features
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction_proba = model.predict_proba(text_vectorized)[0]
        prediction = model.predict(text_vectorized)[0]
        
        # Get confidence
        confidence = float(np.max(prediction_proba))
        
        return PredictionResponse(
            prediction="spam" if prediction == 1 else "ham",
            confidence=confidence,
            is_spam=bool(prediction)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)