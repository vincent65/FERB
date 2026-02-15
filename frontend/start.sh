#!/bin/bash
# EcholoKernel Frontend - Start both backend API and Next.js frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================"
echo "  EcholoKernel — Triton Kernel Optimizer Dashboard"
echo "================================================"
echo ""

# Start FastAPI backend
echo "[1/2] Starting FastAPI backend on port 8000..."
cd "$PROJECT_ROOT"
uvicorn frontend.api.main:app --port 8000 --host 0.0.0.0 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 2

# Start Next.js frontend
echo "[2/2] Starting Next.js frontend on port 3000..."
cd "$SCRIPT_DIR/app"
npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "================================================"
echo "  Dashboard: http://localhost:3000"
echo "  API:       http://localhost:8000/api/health"
echo "================================================"
echo ""
echo "Press Ctrl+C to stop both servers"

# Trap Ctrl+C and clean up
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for either process to exit
wait
