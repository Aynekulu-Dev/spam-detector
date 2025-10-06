from fastapi import FastAPI
from pydantic import BaseModel
import re

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_spam: bool

def simple_spam_detector(text):
    """Simple rule-based spam detection"""
    spam_keywords = [
        'win', 'free', 'money', 'prize', 'congratulations', 'selected',
        'lottery', 'claim', 'urgent', 'suspended', 'selected', 'gift card',
        'limited time', 'buy now', 'rich quick', 'act now'
    ]
    
    text_lower = text.lower()
    spam_score = 0
    
    for keyword in spam_keywords:
        if keyword in text_lower:
            spam_score += 1
    
    # Simple scoring
    word_count = len(text.split())
    spam_ratio = spam_score / max(word_count, 1)
    
    is_spam = spam_ratio > 0.2  # Threshold
    confidence = min(spam_ratio * 2, 0.95)  # Convert to confidence
    
    return is_spam, confidence

@app.get("/")
async def root():
    return {"message": "Spam Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "spam-detection"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if not request.text.strip():
            return PredictionResponse(
                prediction="ham",
                confidence=0.5,
                is_spam=False
            )
        
        is_spam, confidence = simple_spam_detector(request.text)
        
        return PredictionResponse(
            prediction="spam" if is_spam else "ham",
            confidence=confidence,
            is_spam=is_spam
        )
        
    except Exception as e:
        return PredictionResponse(
            prediction="ham",
            confidence=0.5,
            is_spam=False
        )

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)