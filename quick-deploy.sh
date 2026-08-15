#!/bin/bash
# Quick deployment script for exueed.com
# Run this on your GCP VM after initial setup

set -e

echo "🚀 Deploying Paxis..."
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with your API keys"
    exit 1
fi

# Stop existing containers
echo "📦 Stopping existing containers..."
docker-compose down || true

# Pull latest images (if using pre-built)
# docker-compose pull

# Build and start
echo "🔨 Building and starting services..."
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 5

# Check health
echo "🏥 Checking API health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy!"
else
    echo "⚠️  API health check failed. Check logs: docker-compose logs api"
fi

echo ""
echo "=========================================="
echo "✅ Deployment complete!"
echo "=========================================="
echo "Frontend: http://$(curl -s ifconfig.me)"
echo "API: http://$(curl -s ifconfig.me):8000"
echo "API Docs: http://$(curl -s ifconfig.me):8000/docs"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
echo "=========================================="
