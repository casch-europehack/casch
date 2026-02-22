#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Start FastAPI server in the background
echo "[start.sh] Starting FastAPI server..."
python server.py &
SERVER_PID=$!

# Wait for the server to be ready
echo "[start.sh] Waiting for server to be ready..."
until curl -s http://localhost:8000/docs > /dev/null 2>&1; do
    sleep 0.5
done
echo "[start.sh] Server is up (PID $SERVER_PID)"

# Start ngrok tunnel in the background
echo "[start.sh] Starting ngrok tunnel..."
ngrok http 8000 --log=stdout --log-format=json > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok to establish the tunnel and extract the public URL
echo "[start.sh] Waiting for ngrok tunnel..."
PUBLIC_URL=""
for i in $(seq 1 20); do
    sleep 1
    PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
        | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tunnels = data.get('tunnels', [])
    for t in tunnels:
        if t.get('proto') == 'https':
            print(t['public_url'])
            break
except:
    pass
" 2>/dev/null)
    if [ -n "$PUBLIC_URL" ]; then
        break
    fi
done

if [ -z "$PUBLIC_URL" ]; then
    echo "[start.sh] ERROR: Could not get ngrok public URL. Check /tmp/ngrok.log"
    kill $SERVER_PID $NGROK_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=============================================="
echo "  Public URL: $PUBLIC_URL"
echo "  Docs:       $PUBLIC_URL/docs"
echo "  Analyze:    POST $PUBLIC_URL/analyze"
echo "  CO2:        GET  $PUBLIC_URL/co2"
echo "=============================================="
echo ""
echo "[start.sh] Press Ctrl+C to stop everything."

# Shut down both processes cleanly on Ctrl+C
trap "echo ''; echo '[start.sh] Shutting down...'; kill $SERVER_PID $NGROK_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Keep the script alive
wait $SERVER_PID
