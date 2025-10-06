from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import re

app = FastAPI(
    title="Spam Detection API",
    description="Simple spam detection service",
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

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_spam: bool

def clean_text(text):
    """Basic text cleaning"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def simple_spam_detector(text):
    """
    Simple rule-based spam detection
    This works without scikit-learn as a fallback
    """
    spam_keywords = [
        'win', 'free', 'money', 'prize', 'congratulations', 'selected',
        'lottery', 'claim', 'urgent', 'suspended', 'selected', 'gift card',
        'limited time', 'buy now', 'rich quick', 'act now', 'click here',
        'discount', 'offer', 'winner', 'cash', 'bonus', 'reward'
    ]
    
    text_lower = text.lower()
    spam_score = 0
    
    for keyword in spam_keywords:
        if keyword in text_lower:
            spam_score += 1
    
    # Calculate spam probability
    word_count = len(text.split())
    if word_count == 0:
        return False, 0.1
    
    spam_ratio = spam_score / max(word_count, 1)
    
    # More sophisticated scoring
    if spam_score >= 3:
        is_spam = True
        confidence = min(0.3 + (spam_ratio * 0.7), 0.95)
    elif spam_score >= 2:
        is_spam = True
        confidence = min(0.2 + (spam_ratio * 0.6), 0.85)
    elif spam_score >= 1:
        is_spam = spam_ratio > 0.15
        confidence = min(0.1 + (spam_ratio * 0.5), 0.75)
    else:
        is_spam = False
        confidence = max(0.7 - (len(text) / 1000), 0.3)  # Longer legitimate texts get higher confidence
    
    return is_spam, confidence

@app.on_event("startup")
async def startup_event():
    """Initialize the service"""
    print("🚀 Spam Detection API starting up...")
    print("✅ Service ready!")

@app.get("/")
async def root():
    return {
        "message": "Spam Detection API", 
        "status": "running",
        "version": "1.0.0",
        "model_type": "rule-based"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "spam-detection",
        "ready": True
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if not request.text or not request.text.strip():
            return PredictionResponse(
                prediction="ham",
                confidence=0.5,
                is_spam=False
            )
        
        # Use simple rule-based detection
        is_spam, confidence = simple_spam_detector(request.text)
        
        return PredictionResponse(
            prediction="spam" if is_spam else "ham",
            confidence=round(confidence, 3),
            is_spam=is_spam
        )
        
    except Exception as e:
        # Fallback response
        return PredictionResponse(
            prediction="ham",
            confidence=0.5,
            is_spam=False
        )

@app.get("/batch_predict")
async def batch_predict(texts: list[str] = None):
    """Batch prediction endpoint"""
    if not texts:
        return {"predictions": []}
    
    results = []
    for text in texts:
        is_spam, confidence = simple_spam_detector(text)
        results.append({
            "text": text,
            "prediction": "spam" if is_spam else "ham",
            "confidence": round(confidence, 3),
            "is_spam": is_spam
        })
    
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")