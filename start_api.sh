#!/bin/bash
# Free port 8000 and start the Paxis API (so you never get "Address already in use")
cd "$(dirname "$0")"
echo "Freeing port 8000..."
lsof -i :8000 -t 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1
echo "Starting API..."
if command -v python3 &>/dev/null; then
    python3 run_api.py
elif command -v python &>/dev/null; then
    python run_api.py
else
    echo "Error: python3 or python not found. Install Python 3 and try again."
    exit 1
fi
