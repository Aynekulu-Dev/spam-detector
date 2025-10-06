from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import logging
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Spam Detection API",
    description="Machine Learning API for spam detection",
    version="1.0.0"
)

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

@app.on_event("startup")
async def load_model():
    """Load the ML model when the application starts"""
    global model, vectorizer
    try:
        model_path = 'models/spam_model.pkl'
        vectorizer_path = 'models/vectorizer.pkl'
        
        logger.info("🔍 Loading ML model...")
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            logger.warning("❌ Model files not found. Training new model...")
            from utils import train_model
            model, vectorizer = train_model()
        else:
            # Load existing model
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            logger.info("✅ ML model loaded successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        # Don't raise exception, allow service to start without model
        logger.info("🔄 Service starting without model - will train on first request")

@app.get("/")
async def root():
    return {
        "message": "Spam Detection API", 
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict", 
            "batch_predict": "/batch_predict"
        }
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
            # Try to load model if not loaded
            await load_model()
            if model is None:
                raise HTTPException(status_code=503, detail="Model not loaded. Please try again.")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Clean and preprocess text
        cleaned_text = clean_text(request.text)
        
        # Transform text to features
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction_proba = model.predict_proba(text_vectorized)[0]
        prediction = model.predict(text_vectorized)[0]
        
        # Get confidence (probability of the predicted class)
        confidence = float(np.max(prediction_proba))
        
        return PredictionResponse(
            prediction="spam" if prediction == 1 else "ham",
            confidence=confidence,
            is_spam=bool(prediction)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/batch_predict")
async def batch_predict(texts: List[str]):
    """Endpoint for batch predictions"""
    try:
        if model is None or vectorizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
            
        results = []
        for text in texts:
            cleaned_text = clean_text(text)
            text_vectorized = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_vectorized)[0]
            confidence = float(np.max(model.predict_proba(text_vectorized)[0]))
            
            results.append({
                "text": text,
                "prediction": "spam" if prediction == 1 else "ham",
                "confidence": confidence,
                "is_spam": bool(prediction)
            })
        
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)