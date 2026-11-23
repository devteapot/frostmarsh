#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

BIND="${BIND:-127.0.0.1:4433}"
BOTS="${BOTS:-6}"
BACKEND="${BACKEND:-onnx}"
CUDA_ADDR="${CUDA_ADDR:-100.116.218.113:4440}"

echo "[1/2] Starting Zero-WAL QUIC Neural World Server on $BIND (backend=$BACKEND, bots=$BOTS) ..."
./target/release/frostmarsh-server \
  --bind "$BIND" \
  --backend "$BACKEND" \
  --cuda-addr "$CUDA_ADDR" \
  --model "$ROOT_DIR/models/world_model.onnx" \
  --ecosystem-model "$ROOT_DIR/models/ecosystem_model.onnx" \
  --bots "$BOTS" \
  >/tmp/frostmarsh-quic-server.log 2>&1 &
SRV_PID=$!
trap 'kill ${SRV_PID:-} 2>/dev/null || true' EXIT

sleep 1.0
echo "[2/2] Launching Thin QUIC Client + Neural Frame Renderer (A+ v2 / v1) ..."
./target/release/neural-client \
  --server "$BIND" \
  --renderer-v2 "$ROOT_DIR/models/renderer_v2.onnx" \
  --renderer-v1 "$ROOT_DIR/models/renderer_model.onnx" \
  --name "Hero" --team 0
