#!/bin/bash
set -e  # Exit immediately if any command fails

echo "=========================================="
echo "🚀 Starting Medical Chatbot Deployment"
echo "=========================================="

# Step 1: Create embeddings
echo ""
echo "📊 Step 1: Creating embeddings..."
echo "------------------------------------------"
python create_embeddings.py

if [ $? -eq 0 ]; then
    echo "✅ Embeddings created successfully!"
else
    echo "❌ Failed to create embeddings"
    exit 1
fi

# Step 2: Start the Flask application with Gunicorn
echo ""
echo "🌐 Step 2: Starting Flask server..."
echo "------------------------------------------"

# Use gunicorn for production (better than python app.py)
exec gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    --log-level info