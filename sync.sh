#!/bin/bash
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/Users/bytedance/Desktop/hehe/research/rec_sim"
SERVER="http://localhost:8900"

echo "Creating remote directory structure..."
curl -s -X POST "$SERVER/cmd" \
  -H "Content-Type: application/json" \
  -d "{\"cmd\": \"mkdir -p $REMOTE_DIR/src/rec_sim/baseline $REMOTE_DIR/src/rec_sim/fidelity $REMOTE_DIR/src/rec_sim/persona $REMOTE_DIR/src/rec_sim/interaction $REMOTE_DIR/tests\"}"

echo "Directory structure created."
echo "For bulk sync use: scp -r src/ tests/ pyproject.toml <mac>:$REMOTE_DIR/"
