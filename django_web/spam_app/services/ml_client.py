def health_check(self):
    """Check if ML service is healthy"""
    try:
        response = requests.get(f"{self.base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Handle both old and new response formats
            return data.get('status') == 'healthy' or data.get('status') == 'running'
        return False
    except:
        return False