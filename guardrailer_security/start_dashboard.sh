#!/bin/bash
# Guardrailer Launch Script
# Starts both the backend engine and Streamlit dashboard

set -e

echo "============================================"
echo "  Guardrailer - Prompt Security Analyzer"
echo "============================================"
echo ""

# Check if Qdrant is running
echo "Checking Qdrant status..."
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant is running"
else
    echo "⚠️  Qdrant is not running. Starting..."
    docker start guardrailer-qdrant 2>/dev/null || \
    docker run -d --name guardrailer-qdrant -p 6333:6333 -p 6334:6334 \
        -v ./qdrant_storage:/qdrant/storage qdrant/qdrant:latest
    sleep 5
    echo "✅ Qdrant started"
fi

echo ""

# Check if backend is running
echo "Checking backend status..."
if curl -s http://localhost:8090/health > /dev/null 2>&1; then
    echo "✅ Backend is running"
else
    echo "⚠️  Backend is not running. Starting..."
    cd "$(dirname "$0")"
    GUARDRAILER_COLLECTION=guardrailer_security_enhanced python3 security_engine.py &
    BACKEND_PID=$!
    sleep 10
    echo "✅ Backend started (PID: $BACKEND_PID)"
fi

echo ""
echo "============================================"
echo "  Starting Streamlit Dashboard"
echo "============================================"
echo ""
echo "📊 Dashboard will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start Streamlit
cd "$(dirname "$0")"
streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
