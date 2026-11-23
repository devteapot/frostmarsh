#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HF_REPO="${FROSTMARSH_HF_REPO:-carlid/frostmarsh-neural-world}"
MODE="dry-run"
COMMIT_MSG="Update Frostmarsh checkpoints, metrics, and Model Card"
STAGE_DIR="$ROOT_DIR/hf_staging"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift ;;
    --publish) MODE="publish"; shift ;;
    --repo) HF_REPO="$2"; shift 2 ;;
    --commit-message) COMMIT_MSG="$2"; shift 2 ;;
    *) echo "Usage: $0 [--dry-run | --publish] [--repo owner/name] [--commit-message msg]" >&2; exit 2 ;;
  esac
done

echo "=== Frostmarsh Eval Gate & HF Release ($MODE -> $HF_REPO) ==="

python3 - <<'PY'
import json
from pathlib import Path

bot = json.loads(Path("models/BOT_POLICY_METRICS.json").read_text())
shared = bot["shared_latent_bot"]
blind = bot["blind_ablation_bot"]
assert float(shared["val_physics_combo_elem_acc_pct"]) >= 90.0
assert float(shared["val_wildfire_evasion_cosine"]) >= 0.65
assert float(shared["val_physics_combo_elem_acc_pct"]) > float(blind["val_physics_combo_elem_acc_pct"])

unified = json.loads(Path("models/unified_neural_bots/unified_metrics.json").read_text())
assert unified["mass_error"] == 0 and unified["energy_error"] == 0
print(f"[PASS] Gates: combo={shared['val_physics_combo_elem_acc_pct']:.2f}%, wildfire={shared['val_wildfire_evasion_cosine']:.4f}, mass_err=0, energy_err=0")

for opt in ("models/BOT_RL_TOURNAMENT.json", "models/SPARSE_WORLD_BENCH.json"):
    p = Path(opt)
    if p.is_file():
        d = json.loads(p.read_text())
        assert isinstance(d, dict) and len(d) > 0
        print(f"[PASS] Optional metrics validated: {opt}")
PY

EXPECTED=(
  "models/checkpoints/world_model.pt"
  "models/checkpoints/ecosystem_model.pt"
  "models/checkpoints/bot_policy.pt"
  "models/world_model.onnx"
  "models/ecosystem_model.onnx"
)
MISSING=0
for ckpt in "${EXPECTED[@]}"; do
  if [ -s "$ROOT_DIR/$ckpt" ]; then
    echo "  [FOUND]   $ckpt"
  else
    echo "  [MISSING] $ckpt"
    MISSING=$((MISSING + 1))
  fi
done

if [ "$MODE" = "publish" ] && [ "$MISSING" -gt 0 ]; then
  echo "Error: $MISSING required checkpoints missing; run ./scripts/download_models.sh first." >&2
  exit 1
fi

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/checkpoints" "$STAGE_DIR/tiers" "$STAGE_DIR/metrics" "$STAGE_DIR/assets"

cp models/checkpoints/*.pt "$STAGE_DIR/checkpoints/" 2>/dev/null || true
cp models/*.onnx "$STAGE_DIR/" 2>/dev/null || true
cp models/tiers/*.onnx "$STAGE_DIR/tiers/" 2>/dev/null || true
cp models/*.json ml/gate_results.json ml/world_train_summary.json ml/eco_eval_summary.json models/tiers/*.json models/unified_iter2/unified_metrics.json "$STAGE_DIR/metrics/" 2>/dev/null || true
cp models/unified_iter2/unified_live_1080p_hud.png models/acoustic_field_viz.png models/tiers/tier3_hybrid_comparison.png models/tiers/tier1_showcase.png models/tiers/tier4_multistyle.png "$STAGE_DIR/assets/" 2>/dev/null || true

if [ "$MODE" = "dry-run" ]; then
  echo "[DRY-RUN] Staged files in $STAGE_DIR:"
  find "$STAGE_DIR" -type f | sort
  rm -rf "$STAGE_DIR"
  echo "[DRY-RUN PASSED]"
  exit 0
fi

hf upload "$HF_REPO" "$STAGE_DIR" . --repo-type model --commit-message "$COMMIT_MSG"
rm -rf "$STAGE_DIR"
echo "[PUBLISHED] https://huggingface.co/$HF_REPO"
