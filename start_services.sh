#!/usr/bin/env bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
#  start_services.sh
#  Usage: bash start_services.sh
# ─────────────────────────────────────────────────────────────────────────────

# 1. Start MongoDB
echo "▶ Starting MongoDB..."
if command -v systemctl &>/dev/null; then
  sudo systemctl start mongodb || sudo systemctl start mongod
else
  echo "  systemctl not found — assuming MongoDB is already running."
fi

# Wait until MongoDB responds
for i in {1..10}; do
  mongosh --eval "db.adminCommand('ping')" --quiet &>/dev/null && break
  echo "  Waiting for MongoDB ($i/10)..."
  sleep 2
done
echo "  ✓ MongoDB ready."

# 2. Kill stale Python GPU processes
echo "▶ Cleaning GPU..."
nvidia-smi 2>/dev/null \
  | grep 'python' \
  | awk '{ print $5 }' \
  | xargs -r -n1 kill -9 2>/dev/null \
  || true

# 3. Copy .env if it doesn't exist
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  ⚠  Created .env from .env.example — fill in HF_TOKEN before continuing."
  exit 1
fi

# 4. Activate conda env
CONDA_ENV="medgemma"
echo "▶ Activating conda env: $CONDA_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# 5. Start FastAPI backend in background
echo "▶ Starting FastAPI backend on :8000..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# 6. Start Vite frontend
echo "▶ Starting Vite frontend on :5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅  Services running:"
echo "   Backend  → http://localhost:8000"
echo "   API Docs → http://localhost:8000/docs"
echo "   Frontend → http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services."

# Trap Ctrl+C and kill both processes
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" SIGINT SIGTERM
wait
