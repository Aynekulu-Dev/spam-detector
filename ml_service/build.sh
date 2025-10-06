#!/bin/bash
echo "🔧 Building ML Service..."

# Install dependencies using pre-built wheels
pip install -r requirements.txt

# Create models directory
mkdir -p models

echo "🚀 ML Service build completed - model will be loaded at runtime"