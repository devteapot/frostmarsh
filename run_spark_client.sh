#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

SPARK_SERVER="${SPARK_SERVER:-127.0.0.1:4433}"
echo "Connecting Thin QUIC Neural Client to Frostmarsh server at $SPARK_SERVER ..."
exec ./target/release/neural-client \
  --server "$SPARK_SERVER" \
  --renderer-v2 "$ROOT_DIR/models/renderer_v2.onnx" \
  --renderer-v1 "$ROOT_DIR/models/renderer_model.onnx" \
  --name "${PLAYER_NAME:-Hero}" --team "${TEAM:-0}"
