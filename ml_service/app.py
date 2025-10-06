from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

app = FastAPI(
    title="Spam Detection API",
    description="Fast and lightweight spam detection service",
    version="2.0.0"
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

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_type: str

def clean_text(text):
    """Basic text cleaning"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_spam(text):
    """
    Advanced rule-based spam detection
    No machine learning dependencies - pure Python logic
    """
    if not text or not text.strip():
        return False, 0.1
    
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    
    if not words:
        return False, 0.1
    
    # Comprehensive spam patterns
    spam_patterns = [
        # Financial spam
        r'\b(win|won|winner|prize|reward|cash|money|free|bonus)\b',
        r'\b(million|billion|dollar|euro|pound)\b',
        r'\b(rich|wealth|fortune|lottery|jackpot)\b',
        
        # Urgency and pressure
        r'\b(urgent|immediate|instant|limited|quick|fast)\b',
        r'\b(act now|click here|buy now|order now)\b',
        r'\b(discount|offer|deal|sale|clearance)\b',
        
        # Suspicious claims
        r'\b(guarantee|guaranteed|promise|risk.free)\b',
        r'\b(selected|chosen|lucky|exclusive|special)\b',
        r'\b(100% free|no cost|no fee|no obligation)\b',
        
        # Technical spam
        r'\b(account|password|verify|confirm|suspend)\b',
        r'\b(click|link|website|url|http|www)\b',
        
        # Emotional manipulation
        r'\b(congratulation|congrats|amazing|incredible)\b',
        r'\b(opportunity|chance|offer|limited.time)\b'
    ]
    
    # Count spam indicators
    spam_score = 0
    total_words = len(words)
    
    for pattern in spam_patterns:
        matches = re.findall(pattern, cleaned_text)
        spam_score += len(matches)
    
    # Calculate spam probability using sophisticated rules
    spam_ratio = spam_score / max(total_words, 1)
    
    # Advanced scoring algorithm
    if total_words < 3:
        base_score = 0.1
    elif total_words > 50:
        base_score = 0.3  # Long messages are often legitimate
    else:
        base_score = 0.2
    
    # Adjust score based on spam indicators
    if spam_score >= 4:
        is_spam = True
        confidence = min(0.85, base_score + (spam_ratio * 0.8))
    elif spam_score >= 3:
        is_spam = True
        confidence = min(0.75, base_score + (spam_ratio * 0.7))
    elif spam_score >= 2:
        is_spam = spam_ratio > 0.25
        confidence = min(0.65, base_score + (spam_ratio * 0.6))
    elif spam_score >= 1:
        is_spam = spam_ratio > 0.3
        confidence = min(0.55, base_score + (spam_ratio * 0.5))
    else:
        is_spam = False
        # Legitimate messages get higher confidence if they're reasonable length
        confidence = max(0.6, 0.9 - (total_words / 200))
    
    # Special cases - override logic for obvious spam/ham
    obvious_spam_phrases = [
        'win free money', 'congratulations you won', 'you are selected',
        'claim your prize', 'limited time offer', 'act now before'
    ]
    
    obvious_ham_phrases = [
        'hello how are you', 'meeting tomorrow', 'thanks for your',
        'see you later', 'have a good day', 'what time is'
    ]
    
    text_lower = cleaned_text.lower()
    for phrase in obvious_spam_phrases:
        if phrase in text_lower:
            is_spam = True
            confidence = max(confidence, 0.95)
            break
    
    for phrase in obvious_ham_phrases:
        if phrase in text_lower:
            is_spam = False
            confidence = max(confidence, 0.9)
            break
    
    return is_spam, round(confidence, 3)

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="running",
        service="spam-detection",
        version="2.0.0",
        model_type="rule-based-advanced"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="spam-detection",
        version="2.0.0",
        model_type="rule-based-advanced"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        is_spam, confidence = detect_spam(request.text)
        
        return PredictionResponse(
            prediction="spam" if is_spam else "ham",
            confidence=confidence,
            is_spam=is_spam
        )
        
    except Exception as e:
        # Fallback to safe response
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
        is_spam, confidence = detect_spam(text)
        results.append({
            "text": text,
            "prediction": "spam" if is_spam else "ham",
            "confidence": confidence,
            "is_spam": is_spam
        })
    
    return {"predictions": results}

# Test endpoint to verify the detection logic
@app.get("/test")
async def test_endpoint():
    """Test various spam and ham examples"""
    test_cases = [
        "Win free money now! Click here!",
        "Hello, how are you doing today?",
        "Congratulations! You won a $1000 prize!",
        "Meeting at 3 PM tomorrow in conference room",
        "URGENT: Your account will be suspended",
        "Thanks for your help with the project",
        "Free lottery ticket! Claim now!",
        "What time should we meet for lunch?"
    ]
    
    results = []
    for text in test_cases:
        is_spam, confidence = detect_spam(text)
        results.append({
            "text": text,
            "prediction": "spam" if is_spam else "ham",
            "confidence": confidence,
            "is_spam": is_spam
        })
    
    return {"test_results": results}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")