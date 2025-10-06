def predict(self, text):
    """Send text to ML service for spam prediction"""
    try:
        response = requests.post(
            f"{self.base_url}/predict",
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        # Handle both ML and rule-based responses
        if 'prediction' in result and 'confidence' in result:
            return result
        else:
            # Fallback if response format is different
            return {
                "prediction": "ham",
                "confidence": 0.5,
                "is_spam": False
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"ML Service error: {e}")
        return {
            "error": "Spam detection service is currently unavailable",
            "prediction": "ham", 
            "confidence": 0.5,
            "is_spam": False
        }