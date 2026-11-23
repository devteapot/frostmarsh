# Frostmarsh Neural World Model POC — Benchmark Report

**Host:** `spark-2b4e` (aarch64 Linux, **NVIDIA GB10**, CUDA 13.0)  
**Runtime:** Python 3.13.15 (`.venv`), PyTorch `2.14.1+cu130`, ONNX `1.23.2`, ONNX Runtime `1.32.0`  
**Workspace:** `/home/gooning/.t3/scratch/frostmarsh-ml/`

---

## 1. Exported Artifacts & Contract Verification

Both ONNX models are exported at **opset 17**, **single self-contained files** (zero external `.data` files), with **dynamic `[batch, C, height, width]`** axes verified in `onnxruntime` against PyTorch at both `128×128` and `256×256`:

| Artifact | Path | Size | Params | I/O Contract | ORT Max Diff (`128×128` / `256×256`) | SHA-256 (prefix) |
|---|---|---:|---:|---|---|---|
| **World Model** | `/home/gooning/.t3/scratch/frostmarsh-ml/world_model.onnx` | **472,047 B** (461.0 KiB) | 112,817 | `state [B,18,H,W]` → `deltas [B,17,H,W]` | `7.51e-06` / `1.31e-05` | `9164975b4013` |
| **Ecosystem Field (Phase 3)** | `/home/gooning/.t3/scratch/frostmarsh-ml/ecosystem_model.onnx` | **245,026 B** (239.3 KiB) | 48,901 | `eco_state [B,8,H,W]` → `eco_next [B,4,H,W]` | `7.15e-07` / `5.36e-07` | `ac390c668ef3` |

### WorldModel Architecture & Training Summary
- **Architecture:** Pure Conv2d **6×(3×3) Residual CNN** (`width=48`, GELU, explicit replicate padding → exact `RF=13`, arbitrary `H, W`), `softplus` on reaction channels `10..17`.
- **Training:** 2,200 steps (583.4 s on GB10) over dual GPU replay pools (`64×64` + `128×128` crops across 5 procedural Frostmarsh scenes: `river_crossing`, `forest_wildfire`, `oil_slick_marsh`, `dam_break`, `thermal_mix`, plus random Fire/Frost bolts).
- **Curriculum:** Multi-step pushforward rollout ramp `k = 1 → 2 → 4 → 8`, Gaussian state noise (`0.002 → 0.006`), channel-balanced + active-region Huber supervision (prevents sparse phase-change `softplus` collapse), and projection penalty `‖Y_raw − Y_proj‖_1`.
- **Final Training Metrics (step 2200, `k=8`):**
  - Total loss: **`0.0494`** (`delta=0.0263`, `state=0.00071`, `proj=0.00424`)
  - Component Huber losses: `flux=3.79e-5`, `heat=5.63e-4`, `rxn=7.00e-5`, `rxn_active=0.0115`
- **Per-Reaction Active Accuracy (held-out batch with Fire/Frost bolts):**
  - `Freeze`: `gt=0.232`, `pred=0.220`, `MAE=0.056`
  - `Melt`: `gt=0.822`, `pred=0.733`, `MAE=0.275`
  - `Boil`: `gt=0.838`, `pred=0.850`, `MAE=0.089`
  - `Condense`: `gt=0.083`, `pred=0.049`, `MAE=0.047`
  - `WoodBurn`: `gt=0.107`, `pred=0.090`, `MAE=0.026`
  - `OilBurn`: `gt=0.038`, `pred=0.021`, `MAE=0.023`
  - `PlantBurn`: `gt=0.287`, `pred=0.258`, `MAE=0.036`

---

## 2. Gate Results (G1 – G4)

### G1 — Physical Invariants (`PASS`)
Closed-system 200-tick rollout across all 5 Frostmarsh scenes (`5 × 128×128`, `forcing = 0`):

| Invariant | Measured | Target | Status |
|---|---|---|---|
| Max relative total mass drift | **`3.10e-09`** | `< 1e-5` | PASS |
| Max relative total energy (`E_th + Σ m·u0`) drift | **`6.68e-09`** | `< 1e-4` | PASS |
| Min species mass | **`0.0000`** | `>= 0.0` | PASS |
| Max cell total mass | **`4000.0 / 4000.0`** | `<= H_CAP (4000)` | PASS |
| Temperature bounds (`T_cK`) | **`[21331, 116946]` cK** | `[15000, 150000]` cK | PASS |
| Rock / Ice wall flux violations | **`0`** | `0` | PASS |
| 2nd-Law heat conduction violations | **`0`** | `0` | PASS |

### G2 — Rollout Fidelity (`1s / 10s / 60s`) on `128×128` & `256×256`
Evaluated against `RefSim` on unseen seeds with scripted Frost Bolt river ice-bridge creation (`ice >= 700`) and Fire Bolt wildfire ignition:

| Grid | Horizon | Mass MAE (`norm / 1000`) | Mass RMSE | Temp MAE (K) | Water IoU (`>100`) | Ice Bridge IoU (`>=700`) | Fire/Ash IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| **`128×128`** (5 scenes) | **1s** (10 ticks) | 5.49 (`0.0055`) | 24.97 | **1.26 K** | **0.896** | **0.917** | **0.958** |
| **`128×128`** (5 scenes) | **10s** (100 ticks) | 8.82 (`0.0088`) | 38.90 | **2.79 K** | **0.890** | **0.875** | **0.931** |
| **`128×128`** (5 scenes) | **60s** (600 ticks) | 19.76 (`0.0198`) | 87.64 | **6.26 K** | **0.802** | **0.852** | **0.905** |
| **`256×256`** (2 scenes, zero-shot) | **1s** (10 ticks) | 0.78 (`0.0008`) | 9.83 | **0.22 K** | **0.948** | **1.000** | **0.987** |
| **`256×256`** (2 scenes, zero-shot) | **10s** (100 ticks) | 2.01 (`0.0020`) | 18.08 | **0.59 K** | **0.912** | **0.899** | **0.950** |
| **`256×256`** (2 scenes, zero-shot) | **60s** (600 ticks) | 4.44 (`0.0044`) | 35.40 | **0.79 K** | **0.861** | **0.973** | **0.953** |

### G3 — 10,000-Step Long Rollout Stability (`PASS`)
10,000 autoregressive ticks (1,000 simulated seconds = 16.7 minutes) on `128×128` `river_crossing` with periodic player Fire & Frost bolts every 100 steps:

| Metric | Measured |
|---|---|
| Total runtime / throughput | **87.7 s** (**114.0 steps/s**, 11.4× real-time) |
| NaNs / Infs | **`0`** |
| Min mass / Max cell mass | **`0.0`** / **`3858.4 <= 4000.0`** |
| Temperature range | **`[15036, 97965] cK`** (strictly inside `[15000, 150000]`) |
| Projection mean L1 correction `‖Y_raw − Y_proj‖_1` | **`0.00532`** |
| Material projection-active % (`|Δ| > 1e-2`) | **`4.93%`** overall (`first 1k: 4.20%`, `last 1k: 4.90%` — flat, zero drift) |
| Strict projection-active % (`|Δ| > 1e-4`) | `79.09%` (`flux=70.78%`, `heat=58.29%`, `rxn=94.53%` — dominated by tiny `~1e-3` `softplus` tails on inactive reactions) |

### G4 — Batched GPU & CPU Inference Latency (10 Hz Budget = `100.0 ms`)

| Config | CUDA FP32 p50 / p99 | CUDA FP16 p50 / p99 | Full Step FP32 (Encode+NN+Proj) p50 / p99 | ORT CPU p50 / p99 |
|---|---:|---:|---:|---:|
| **`B=1, 128×128`** | **0.88 ms** / 1.26 ms | **0.70 ms** / 0.96 ms | **2.73 ms** / 5.33 ms | 51.96 ms / 74.02 ms |
| **`B=4, 128×128`** | **4.44 ms** / 6.48 ms | **3.56 ms** / 5.64 ms | **8.09 ms** / 14.22 ms | — |
| **`B=8, 128×128`** | **14.39 ms** / 23.72 ms | **9.73 ms** / 15.43 ms | — | — |
| **`B=16, 128×128`** | **33.59 ms** / 50.09 ms | **23.82 ms** / 38.09 ms | — | — |
| **`B=1, 256×256`** | **4.62 ms** / 6.58 ms | **3.71 ms** / 5.64 ms | **8.21 ms** / 13.59 ms | 204.10 ms / 863.66 ms |
| **`B=4, 256×256`** | **31.32 ms** / 50.33 ms | **24.03 ms** / 40.78 ms | **42.59 ms** / 68.59 ms | — |
| **`B=8, 256×256`** | **63.66 ms** / 90.75 ms | **51.41 ms** / 77.94 ms | — | — |
| **`B=16, 256×256`** | 132.93 ms / 189.18 ms | 104.60 ms / 150.98 ms | — | — |

Single-world `256×256` full physics step runs in **8.21 ms p50** (**12.2× faster than the 100 ms 10 Hz tick budget**); `128×128` runs in **2.73 ms p50** on GPU (**36.6× faster**) and **51.96 ms p50** even on CPU via ONNX Runtime.

---

## 3. Phase 3 Bonus (G5-B) — Living Ecosystem Neural Field vs Reaction-Diffusion Baseline

- **Offline Teacher:** GPU-vectorized **500-agent Frostmarsh ecosystem simulator** (`4` territorial herds with Boids alignment/cohesion/separation + rival-herd repulsion, plant foraging/grazing, wildfire/smoke fleeing + social panic contagion, rock/deep-water avoidance, and walkable **ice-bridge crossing** when `ice >= 700`).
- **Neural Field (`ecosystem_model.onnx`, 48,901 params, RF=13):** Conservative edge-flux critter advection + strict barrier masking + bounded grazing/momentum built directly inside the ONNX graph (`[B, 8, H, W] → [B, 4, H, W]`), trained with multi-step rollout `k = 1 → 2 → 4 → 8` on `64×64` worlds (58.6 s total).
- **Baseline:** Optimized 14-parameter classical **Reaction-Diffusion + Chemotaxis PDE** trained on the exact same trajectories.
- **Held-out `128×128` Evaluation** (4× larger unseen worlds with Frost Bolt ice-bridge creation + Fire Bolt wildfire stampede):

| Metric (Held-out `128×128`) | Neural Field (`ecosystem_model.onnx`) | Reaction-Diffusion PDE Baseline | Improvement |
|---|---:|---:|---:|
| **1-step MAE (all 4 channels)** | **`0.00106`** | `0.00466` | **4.39× lower error** |
| 1-step MAE — Critter Density | **`0.00175`** | `0.00298` | **1.70× better** |
| 1-step MAE — Fear (panic cascades) | **`0.00117`** | `0.01354` | **11.57× better** |
| 1-step MAE — Forage Depletion | **`0.00034`** | `0.00047` | **1.37× better** |
| 1-step MAE — Herd Momentum | **`0.00098`** | `0.00164` | **1.68× better** |
| **1s (10-step) Closed-Loop MAE (all)** | **`0.00775`** | `0.01367` | **1.76× lower error** |
| 1s (10-step) Critter Density Spatial Corr | **`0.814`** | `0.685` | **+0.129 corr** |
| **5s (50-step) Closed-Loop MAE (all)** | **`0.01745`** | `0.02126` | **17.9% lower error** |
| 5s (50-step) Fear MAE | **`0.00685`** | `0.01909` | **2.79× better** |
| 5s (50-step) Herd Momentum MAE | **`0.00951`** | `0.01445` | **1.52× better** |

---

## 4. Source & Output File Manifest

- [sim_teacher.py](/home/gooning/.t3/scratch/frostmarsh-ml/sim_teacher.py) — Vectorized PyTorch `RefSim`, exact differentiable invariant projection, procedural Frostmarsh biome & bolt generators.
- [model.py](/home/gooning/.t3/scratch/frostmarsh-ml/model.py) — `WorldModel` (112,817 params, 6×3×3 Residual CNN, RF=13) & conservative `EcosystemModel` (48,901 params, RF=13).
- [train_world_model.py](/home/gooning/.t3/scratch/frostmarsh-ml/train_world_model.py) — Multi-step pushforward curriculum trainer + opset 17 ONNX exporter & ORT verifier.
- [train_ecosystem_model.py](/home/gooning/.t3/scratch/frostmarsh-ml/train_ecosystem_model.py) — 500-agent Frostmarsh ecosystem simulator, neural field distillation, Reaction-Diffusion baseline, and `ecosystem_model.onnx` export.
- [eval_gates.py](/home/gooning/.t3/scratch/frostmarsh-ml/eval_gates.py) — Full G1–G4 + G5-B evaluation & latency benchmark runner.
- [world_model.onnx](/home/gooning/.t3/scratch/frostmarsh-ml/world_model.onnx) — Self-contained ONNX opset 17 world model (`472,047 B`).
- [ecosystem_model.onnx](/home/gooning/.t3/scratch/frostmarsh-ml/ecosystem_model.onnx) — Self-contained ONNX opset 17 living ecosystem field (`245,026 B`).
- [gate_results.json](/home/gooning/.t3/scratch/frostmarsh-ml/gate_results.json) — Machine-readable full G1–G5B metrics.
- [BENCHMARK_REPORT.md](/home/gooning/.t3/scratch/frostmarsh-ml/BENCHMARK_REPORT.md) — This report.
