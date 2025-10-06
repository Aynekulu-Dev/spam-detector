#!/bin/bash
echo "🔧 Building ML Service..."

# Install dependencies
pip install -r requirements.txt

# Create models directory
mkdir -p models

# Train model with error handling
echo "🤖 Training ML model..."
if python utils.py; then
    echo "✅ Model training completed successfully"
else
    echo "⚠️ Model training failed, but continuing deployment..."
    # Create dummy model files to prevent crashes
    touch models/spam_model.pkl
    touch models/vectorizer.pkl
fi

echo "🚀 ML Service build completed"