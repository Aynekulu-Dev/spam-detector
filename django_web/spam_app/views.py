from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import logging

# Import the ML client - we'll create this next
try:
    from .services.ml_client import ml_client
except ImportError:
    # Create a dummy client for now
    class DummyMLClient:
        def predict(self, text):
            return {"error": "ML client not configured"}
        def health_check(self):
            return False
    ml_client = DummyMLClient()

# Import models
try:
    from .models import Prediction
except ImportError:
    # Define a simple Prediction class if models aren't available yet
    class Prediction:
        objects = type('Objects', (), {'all': lambda self: [], 'create': lambda self, **kwargs: type('Obj', (), kwargs)()})()
        @classmethod
        def objects(cls):
            return type('Objects', (), {
                'all': lambda self: [],
                'order_by': lambda self, order: self,
                '__getitem__': lambda self, index: []
            })()

logger = logging.getLogger(__name__)

def home(request):
    """Render the main page"""
    service_status = ml_client.health_check()
    
    try:
        recent_predictions = Prediction.objects.all()[:5]
    except:
        recent_predictions = []
    
    return render(request, 'home.html', {
        'service_status': service_status,
        'recent_predictions': recent_predictions
    })

@require_http_methods(["POST"])
@csrf_exempt
def check_spam(request):
    """API endpoint to check if text is spam"""
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        
        if not text:
            return JsonResponse({'error': 'Text is required'}, status=400)
        
        if len(text) > 1000:
            return JsonResponse({'error': 'Text too long (max 1000 characters)'}, status=400)
        
        # Call ML service
        result = ml_client.predict(text)
        
        if 'error' in result:
            return JsonResponse({'error': result['error']}, status=503)
        
        # Save to database if models are available
        try:
            prediction = Prediction.objects.create(
                text=text,
                prediction=result['prediction'],
                confidence=result['confidence'],
                is_spam=result['is_spam']
            )
            result['id'] = prediction.id
        except:
            result['id'] = 1  # Dummy ID if database not available
        
        return JsonResponse(result)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JsonResponse({'error': 'Internal server error'}, status=500)

@require_http_methods(["GET"])
def prediction_history(request):
    """Get prediction history"""
    try:
        predictions = Prediction.objects.all().order_by('-created_at')[:10]
        data = [
            {
                'id': p.id,
                'text': p.text,
                'prediction': p.prediction,
                'confidence': p.confidence,
                'is_spam': p.is_spam,
                'created_at': p.created_at.isoformat() if hasattr(p.created_at, 'isoformat') else str(p.created_at)
            }
            for p in predictions
        ]
    except:
        data = []
    
    return JsonResponse({'predictions': data})

@require_http_methods(["GET"])
def service_status(request):
    """Check ML service status"""
    status = ml_client.health_check()
    return JsonResponse({
        'ml_service_status': 'online' if status else 'offline',
        'ml_service_url': getattr(ml_client, 'base_url', 'http://localhost:8001')
    })