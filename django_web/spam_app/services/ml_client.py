import requests
from django.conf import settings
import logging
import os

logger = logging.getLogger(__name__)

class MLServiceClient:
    def __init__(self):
        # Try to get ML service URL from different environment variables
        if hasattr(settings, 'ML_SERVICE_URL'):
            self.base_url = settings.ML_SERVICE_URL
        elif os.environ.get('ML_SERVICE_HOST'):
            # Render deployment - construct URL from host and port
            host = os.environ.get('ML_SERVICE_HOST')
            port = os.environ.get('ML_SERVICE_PORT', '10000')
            self.base_url = f"https://{host}"  # Render uses HTTPS
        else:
            # Local development default
            self.base_url = 'http://localhost:8001'
        
        # Ensure no trailing slash
        self.base_url = self.base_url.rstrip('/')
        logger.info(f"ML Service Client initialized with URL: {self.base_url}")
    
    def predict(self, text):
        """Send text to ML service for spam prediction"""
        try:
            url = f"{self.base_url}/predict"
            logger.info(f"Sending request to: {url}")
            
            response = requests.post(
                url,
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"ML Service error: {e} - URL: {self.base_url}")
            return {"error": "Spam detection service is currently unavailable"}
    
    def health_check(self):
        """Check if ML service is healthy"""
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"ML Service health check failed: {e}")
            return False

# Create a singleton instance
ml_client = MLServiceClient()