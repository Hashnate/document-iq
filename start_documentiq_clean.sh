#!/bin/bash
echo "🚀 Starting DocumentIQ (clean, no warnings)..."

# Create directories
mkdir -p logs uploads extracted_files temp_processing

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU:" $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
else
    echo "⚠️  No GPU detected"
fi

echo "Starting with sync workers (recommended)..."
gunicorn --workers 6 \
         --worker-class sync \
         --threads 4 \
         --timeout 7200 \
         --graceful-timeout 7200 \
         --bind 127.0.0.1:8000 \
         --access-logfile ./logs/access.log \
         --error-logfile ./logs/error.log \
         --log-level info \
         --max-requests 50 \
         app:app
