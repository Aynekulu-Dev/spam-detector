
## Services

- **Django Web**: Web interface for spam detection
- **ML Service**: FastAPI service with spam classification model

## Setup

1. **With Docker**: `docker-compose up --build`
2. **Without Docker**: See individual service README files

## API Endpoints

- ML Service: http://localhost:8001
- Django Web: http://localhost:8000
EOL

# Create individual README files for each service
cat > django_web/README.md << 'EOL'
# Django Web Application

Web interface for the spam detection system.

## Features

- Real-time spam detection
- Prediction history
- ML service integration

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver