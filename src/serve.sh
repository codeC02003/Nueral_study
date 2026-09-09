#!/bin/bash
# Snapshot the live networks, rebuild, and serve locally. Nothing leaves this machine.
cd "$(dirname "$0")/.."
python3 viz/export_neurons.py || exit 1
python3 src/build_anatomy.py  || exit 1
PORT=${PORT:-8777}
pkill -f "http.server $PORT" 2>/dev/null
cd viz && nohup python3 -m http.server "$PORT" >/dev/null 2>&1 &
sleep 1
echo
echo "  http://localhost:$PORT/anatomy.html"
echo
command -v open >/dev/null && open "http://localhost:$PORT/anatomy.html"
