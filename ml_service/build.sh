#!/bin/bash
echo "🔧 Building ML Service..."

# Install dependencies
pip install -r requirements.txt

# Create models directory
mkdir -p models

echo "🚀 ML Service build completed"