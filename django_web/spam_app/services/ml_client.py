import requests
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class MLServiceClient:
    def __init__(self):
        self.base_url = getattr(settings, 'ML_SERVICE_URL', 'http://localhost:8001')
    
    def predict(self, text):
        """Send text to ML service for spam prediction"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"text": text},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"ML Service error: {e}")
            return {"error": "Spam detection service is currently unavailable"}
    
    def health_check(self):
        """Check if ML service is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

# Create a singleton instance
ml_client = MLServiceClient()