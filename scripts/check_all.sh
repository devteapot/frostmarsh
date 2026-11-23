#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Frostmarsh Pre-Push Verification Suite ==="
echo "[1/5] Rust conservation unit tests (field-core 2D & 3D)..."
cargo test -p field-core

echo "[2/5] Rust workspace checks (proto, server, unified-client)..."
cargo check -p frostmarsh-proto
cargo check --no-default-features -p frostmarsh-server
cargo check -p frostmarsh-unified-client

echo "[3/5] Python ML syntax check..."
shopt -s nullglob
PY_FILES=(ml/*.py ml/tiers/*.py)
shopt -u nullglob
python3 -m py_compile "${PY_FILES[@]}"
echo "Compiled ${#PY_FILES[@]} Python files cleanly."

echo "[4/5] Validating metric JSON invariant gates..."
python3 - <<'PY'
import json
from pathlib import Path

unified = json.loads(Path("models/unified_neural_bots/unified_metrics.json").read_text())
assert unified["mass_error"] == 0, f"mass_error={unified['mass_error']}"
assert unified["energy_error"] == 0, f"energy_error={unified['energy_error']}"

bot = json.loads(Path("models/BOT_POLICY_METRICS.json").read_text())
shared = bot["shared_latent_bot"]
blind = bot["blind_ablation_bot"]
assert float(shared["val_physics_combo_elem_acc_pct"]) >= 90.0
assert float(shared["val_wildfire_evasion_cosine"]) >= 0.65
assert float(shared["val_physics_combo_elem_acc_pct"]) > float(blind["val_physics_combo_elem_acc_pct"])
assert float(shared["val_wildfire_evasion_cosine"]) > float(blind["val_wildfire_evasion_cosine"])

for opt in ("models/BOT_RL_TOURNAMENT.json", "models/SPARSE_WORLD_BENCH.json"):
    p = Path(opt)
    if p.is_file():
        d = json.loads(p.read_text())
        assert isinstance(d, dict) and len(d) > 0
        print(f"  [OK] {opt} valid")

print("  [OK] Core invariants & policy gates passed!")
PY

echo "[5/5] ALL CHECKS PASSED"
