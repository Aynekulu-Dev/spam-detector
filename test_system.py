#!/usr/bin/env python3
"""
Final system test script
"""
import requests
import time
import sys

def test_ml_service():
    """Test ML service health and predictions"""
    print("🔬 Testing ML Service...")
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ ML Service health: OK")
        else:
            print("❌ ML Service health: FAILED")
            return False
        
        # Test spam prediction
        spam_response = requests.post(
            "http://localhost:8001/predict",
            json={"text": "Win free money now!"},
            timeout=5
        )
        if spam_response.status_code == 200:
            result = spam_response.json()
            print(f"✅ Spam detection: {result['prediction']} ({result['confidence']:.2f})")
        else:
            print("❌ Spam detection: FAILED")
            return False
            
        # Test ham prediction
        ham_response = requests.post(
            "http://localhost:8001/predict",
            json={"text": "Hello, how are you?"},
            timeout=5
        )
        if ham_response.status_code == 200:
            result = ham_response.json()
            print(f"✅ Ham detection: {result['prediction']} ({result['confidence']:.2f})")
        else:
            print("❌ Ham detection: FAILED")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ ML Service test failed: {e}")
        return False

def test_django_service():
    """Test Django web service"""
    print("\n🌐 Testing Django Web Service...")
    try:
        # Test home page
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Django home page: OK")
        else:
            print("❌ Django home page: FAILED")
            return False
        
        # Test API status
        status_response = requests.get("http://localhost:8000/api/status/", timeout=5)
        if status_response.status_code == 200:
            print("✅ Django API status: OK")
        else:
            print("❌ Django API status: FAILED")
            return False
            
        # Test spam detection API
        api_response = requests.post(
            "http://localhost:8000/api/check-spam/",
            json={"text": "Free lottery ticket!"},
            timeout=5
        )
        if api_response.status_code == 200:
            result = api_response.json()
            print(f"✅ Django API spam detection: {result['prediction']}")
        else:
            print("❌ Django API spam detection: FAILED")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Django Service test failed: {e}")
        return False

def main():
    print("🚀 Starting Final System Test...")
    print("Make sure both services are running!")
    print("ML Service: http://localhost:8001")
    print("Django Web: http://localhost:8000")
    print()
    
    time.sleep(2)  # Give user time to read
    
    ml_ok = test_ml_service()
    django_ok = test_django_service()
    
    print("\n" + "="*50)
    if ml_ok and django_ok:
        print("🎉 ALL TESTS PASSED! System is working correctly.")
        print("\nYou can now:")
        print("1. Open http://localhost:8000 in your browser")
        print("2. Test various spam and ham messages")
        print("3. View prediction history")
    else:
        print("❌ SOME TESTS FAILED! Check the services.")
        sys.exit(1)

if __name__ == "__main__":
    main()