#!/bin/bash

# Quick Start Script for Dysarthria Detection App
# Run this after cloning the repository

set -e

echo "🚀 Dysarthria Detection - Quick Start"
echo "===================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker found"
echo "✅ Docker Compose found"

# Create directories
echo "📁 Creating directories..."
mkdir -p backend/models
mkdir -p frontend/src
mkdir -p frontend/public
mkdir -p .github/workflows

# Check model files
if [ ! -f "backend/models/catboost_presence1.cbm" ]; then
    echo "⚠️  Model file catboost_presence1.cbm not found in backend/models/"
    echo "   Please copy your model files:"
    echo "   cp catboost_presence1.cbm backend/models/"
    echo "   cp catboost_severity1.cbm backend/models/"
fi

# Build images
echo "🔨 Building Docker images..."
docker-compose build

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 15

# Check health
echo "🏥 Checking health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is running"
else
    echo "❌ Backend failed to start"
    docker-compose logs backend
    exit 1
fi

echo ""
echo "✅ All services started successfully!"
echo ""
echo "📍 Access your app:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📊 View logs:"
echo "   docker-compose logs -f backend"
echo "   docker-compose logs -f frontend"
echo ""
echo "🛑 Stop services:"
echo "   docker-compose down"
echo ""
echo "🚀 Ready for deployment!"
