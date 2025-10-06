#!/bin/bash
echo "Building ML service..."

# Install dependencies
pip install -r requirements.txt

# Train the model
python utils.py

echo "ML service build completed successfully!"