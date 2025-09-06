#!/bin/bash
# Quick start script for Traffic Violation Detection System

echo "🚀 Starting Traffic Violation Detection System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/installed" ]; then
    echo "📥 Installing requirements..."
    pip install Flask opencv-python numpy Pillow
    echo "⚠️  Installing AI models (optional, can skip if error)..."
    pip install ultralytics easyocr || echo "⚠️  AI models skipped - will use fallback methods"
    touch venv/installed
fi

# Run the application
echo "🌐 Starting web server..."
echo "📱 Access the app at: http://localhost:5000"
python app.py
