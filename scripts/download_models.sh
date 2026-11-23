#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HF_REPO="${FROSTMARSH_HF_REPO:-carlid/frostmarsh-neural-world}"

echo "Downloading Frostmarsh pretrained checkpoints from https://huggingface.co/$HF_REPO ..."
mkdir -p "$ROOT_DIR/models/checkpoints" "$ROOT_DIR/models/tiers"

if command -v hf >/dev/null 2>&1; then
  hf download "$HF_REPO" --local-dir "$ROOT_DIR/models" --include "*.onnx" "*.pt" "checkpoints/*" "tiers/*.onnx"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$HF_REPO" --local-dir "$ROOT_DIR/models" --include "*.onnx" "*.pt" "checkpoints/*" "tiers/*.onnx"
else
  echo "Error: install huggingface_hub CLI first: pip install -U huggingface_hub" >&2
  exit 1
fi

echo "Done! Models ready in $ROOT_DIR/models"
