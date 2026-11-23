# Frostmarsh Tier 3 & Tier 4 — Implementation & Benchmark Report

**Host / Accelerator:** `spark-2b4e` — NVIDIA GB10 GPU (128 GB unified memory, CUDA 13.0, PyTorch `2.14.1+cu130`, Triton `3.8.0`, ONNX `1.23.2`, ONNXRuntime `1.32.0`, Diffusers `0.43.0`)  
**Workspace:** `/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/`  
**Shared Wire Contract:** Exact Frostmarsh 4-byte `A+` wire codec (`decode_aplus_wire` → `[B, 17, 128, 128]` float32: 10 compositional species mass fractions, `temp_norm`, `burning`, `flow_x`, `flow_y`, `critter_density`, `time_sin`, `time_cos`) driven by live 10 Hz `RefSim` physics (`sim_teacher.py`).

---

## Executive Summary

| Pipeline | Core Capability | Measured Headline Result (NVIDIA GB10) | Peak GPU VRAM |
| :--- | :--- | :--- | :---: |
| **Tier 3 — Neural `A+` → Dynamic 3DGS** (`tier3_3dgs.py`) | 10 Hz neural Gaussian generator (`93,860` params, `tier3_gs_gen.onnx`) decoupled from real-time Triton 3DGS camera rasterization | **0.312 ms** full `128²` / **0.163 ms** 15% dirty-chunk (GPU FP16) · **1.43 ms** 15% dirty-chunk (CPU ORT 4t) · **292.8 FPS @ 512×512** & **297.3 FPS @ 256×256** with **ZERO neural inference per camera frame** and **100.0% geometric temporal consistency** | **1,286.6 MB** (1.26 GB) |
| **Tier 4 — Swappable Multi-Style Client Renderer** (`tier4_diffusion.py`) | Real `stabilityai/sd-turbo` (1–4 step adversarial diffusion) rendering 3 radically different AAA client art styles from the exact same Frostmarsh world tick | **87.5 ms (11.4 FPS)** 1-step · **141.4 ms (7.1 FPS)** 2-step · **239.4 ms (4.2 FPS)** 4-step @ `512×512` (FP16 + TAESD fast VAE); UNet-only **40.2 ms (24.9 FPS)** 1-step · Optical-flow + motion-gated latent EMA cuts temporal flicker by **-76.1%** vs raw (**+11.9 dB** temporal PSNR) | **3,385.1 MB** (3.31 GB) |

---

## Part A: Tier 3 — Neural `A+` State (10 Hz) → Dynamic 3D Gaussian Splatting

### 1. Architecture & Differentiable 3DGS Training
- **Model (`GaussianGeneratorNet`):** `93,860` trainable parameters — a purely translation-equivariant CNN (3×3 stem `17→48`, two residual blocks with `dilation=1` and `dilation=2` for 7×7 neighborhood context, 1×1 head `48→20`, `SiLU` activations, zero BatchNorm/global pooling).
- **Primitives Predicted:** `K = 2` 3D Gaussians per cell (`2 × 128 × 128 = 32,768` primitives total; `21,389` active after opacity pruning `α > 0.05`). Each Gaussian predicts 10 parameters: `(dx, dy, z, log_sx, log_sy, log_sz, r, g, b, opacity_logit)`.
- **Bit-Exact Dirty-Chunk Equivariance:** Because the generator uses no spatial normalization or global pooling, running inference on a 15% dirty sub-grid (`48×52` cells) with a 6-cell receptive-field halo produces **bit-identical** Gaussian parameters on the tile interior (`dirty_chunk_interior_equiv_err = 0.00e+00`).
- **End-to-End Differentiable Triton 3DGS Rasterizer:**
  - Full EWA 3D-to-2D Jacobian covariance projection (`Σ_2D = J R diag(s²) Rᵀ Jᵀ + 0.35 I`) with exact pinhole 3D orbit cameras `(yaw, pitch, dist, fov)`.
  - GPU `16×16` tile binning + CUB 64-bit `(tile_id << 32 | depth_u32)` front-to-back radix sort + Triton forward & analytic reverse-order alpha-compositing backward kernels.
  - Trained end-to-end for **1,500 multi-view steps** (`61.27 s`) from random 3D orbit cameras (`yaw ∈ [0°, 360°]`, `pitch ∈ [24°, 62°]`, `dist ∈ [1.65, 2.35]`) against 9 live 128×128 Frostmarsh scenes using `L1 + 0.08 · Sobel + 0.004 · sparsity`.
  - **Held-Out 8-View Orbit PSNR:** **24.04 dB** (hero Frostmarsh battle scene), **22.68 dB** (held-out unseen `river_crossing` scene).

### 2. ONNX Export & Verification
- **Artifacts:** `tier3_gs_gen.onnx` (`369.7 KB`, opset 17, self-contained single file with dynamic `(batch, height, width)` axes) and `tier3_gs_gen.pt`.
- **Verification (`onnx.checker` + ONNXRuntime CPU):**
  - Full `128×128` max abs diff vs PyTorch: **`6.68e-06`**
  - Dirty chunk (`48×52`, 15.2%) max abs diff vs PyTorch: **`5.25e-06`**
  - Dirty-chunk interior equivalence vs full-grid pass: **`0.00e+00`**

### 3. Decoupled Latency & FPS Benchmarks

#### A. 10 Hz Neural Gaussian Update (runs ONLY on server tick — 100 ms budget)

| Workload / Precision / Runtime | Grid Size | Median Latency | p95 Latency | Max Rate | Share of 100 ms (10 Hz) Budget |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **GPU FP16 — Full Grid** | `128×128` (16,384 cells) | **0.312 ms** | 0.509 ms | **3,205.6 Hz** | **0.31%** |
| **GPU FP16 — 15% Dirty Chunk** | `48×52` (2,496 cells) | **0.163 ms** | 0.377 ms | **6,133.5 Hz** | **0.16%** |
| **GPU FP32 — Full Grid** | `128×128` (16,384 cells) | **0.512 ms** | 0.520 ms | **1,953.5 Hz** | **0.51%** |
| **GPU FP32 — 15% Dirty Chunk (Slice)** | `48×52` (2,496 cells) | **0.175 ms** | 0.280 ms | **5,714.0 Hz** | **0.18%** |
| **GPU FP32 — 15% Dirty Chunks (10× `16²` tiles)** | `10×16×16` (2,560 cells) | **0.174 ms** | 0.183 ms | **5,747.1 Hz** | **0.17%** |
| **CPU ONNXRuntime (4 threads) — Full Grid** | `128×128` (16,384 cells) | **9.699 ms** | 10.165 ms | **103.1 Hz** | **9.70%** |
| **CPU ONNXRuntime (4 threads) — 15% Dirty Chunk** | `48×52` (2,496 cells) | **1.430 ms** | 1.467 ms | **699.2 Hz** | **1.43%** |

#### B. 3DGS Camera Rasterization (ZERO Neural Inference per Frame!)

| Render Resolution | Gaussian Count | Median Frame Time | p95 Frame Time | Sustained FPS | Geometric Temporal Consistency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`256×256` (Active Pruned)** | `21,389` | **3.364 ms** | 3.941 ms | **297.3 FPS** | **100.0%** (`0.0` static repeat diff) |
| **`512×512` (Active Pruned)** | `21,389` | **3.416 ms** | 4.095 ms | **292.8 FPS** | **100.0%** (`0.0` static repeat diff) |
| **`256×256` (All 32k Unpruned)** | `32,768` | **3.473 ms** | 3.909 ms | **288.0 FPS** | **100.0%** |
| **`512×512` (All 32k Unpruned)** | `32,768` | **3.568 ms** | 4.159 ms | **280.3 FPS** | **100.0%** |

> **Key Takeaway:** Because the 3D Gaussian cloud updates at 10 Hz (`0.16–0.31 ms` on GPU, or `1.43 ms` dirty-chunk on CPU!) and the camera rasterizer runs purely geometrically at **292.8 FPS @ 512×512**, camera rotation/pan/zoom incurs **zero neural inference** and exhibits **100% view-consistent 3D parallax** (`0.000000` static repeat error; `0.01934` smooth L1 on 0.5° orbit micro-steps).

### 4. Visual Showcase — 4 Live 3D Orbit Views
![Tier 3 4-View 3DGS Orbit](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tier3_3dgs_orbit.png)

---

## Part B: Tier 4 — Few-Step Real-Time Diffusion / Swappable Multi-Style Client Renderer

### 1. Architecture & Swappable Client Art Styles
- **Model:** `stabilityai/sd-turbo` (real 1–4 step adversarial diffusion distilled checkpoint, `float16`, official `EulerDiscreteScheduler` with `trailing` timestep spacing and `guidance_scale=0.0`).
- **Cached Style Conditioning:** Each player's style prompt is CLIP-encoded **once** upon switching art styles (`0.0 ms` per-frame text overhead).
- **Exact Same Server Tick → 3 Radically Different AAA Clients:** Conditioning directly on the same projected `512×512` Tier 3 Frostmarsh battle frame (`battle_frame_02.png` — diagonal glacial river, ice bridge, wildfire, granite boulders, pine forest):
  1. **Style 1 — Dark Fantasy Photoreal** (*gritty dark fantasy RPG terrain, roaring volumetric firelight, wet jagged basalt cliffs, frosted river crossing*)
  2. **Style 2 — Studio Ghibli Painterly** (*hand-painted anime landscape, lush watercolor emerald hills, warm sunlit flames, alpine crags, azure river*)
  3. **Style 3 — Cyberpunk Neon Tactical** (*bioluminescent cyan neon grid, glowing holographic wireframe pads, magenta/orange plasma fire towers, obsidian synthwave board*)

### 2. Real GPU Latency & FPS Benchmarks (`512×512`, FP16, Verified UNet NFE)

Every step configuration (`1`, `2`, `4`) was verified via a PyTorch forward hook on `pipe.unet` to guarantee exact UNet NFE counts (`1`, `2`, `4`). We benchmarked **both** the stock SD `AutoencoderKL` VAE and the production real-time distilled **`madebyollin/taesd`** (`9.8 MB`) VAE:

| Configuration (`512×512`, FP16) | Verified UNet NFE | Median E2E Latency | p95 Latency | Sustained E2E FPS | Component Breakdown |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SD-Turbo + TAESD Fast VAE — 1-Step** | `1` | **87.49 ms** | 102.41 ms | **11.43 FPS** | TAESD enc `13.5 ms` + UNet `~40.2 ms` + TAESD dec `11.4 ms` |
| **SD-Turbo + TAESD Fast VAE — 2-Step** | `2` | **141.38 ms** | 159.22 ms | **7.07 FPS** | TAESD enc `13.5 ms` + 2×UNet `~97.7 ms` + TAESD dec `11.4 ms` |
| **SD-Turbo + TAESD Fast VAE — 4-Step** | `4` | **239.38 ms** | 270.65 ms | **4.18 FPS** | TAESD enc `13.5 ms` + 4×UNet `~206.5 ms` + TAESD dec `11.4 ms` |
| **SD-Turbo + Stock AutoencoderKL — 1-Step** | `1` | **605.47 ms** | 644.70 ms | **1.65 FPS** | Stock VAE enc `193.0 ms` + UNet `40.2 ms` (24.9 FPS) + dec `372.3 ms` |
| **SD-Turbo + Stock AutoencoderKL — 2-Step** | `2` | **662.95 ms** | 687.27 ms | **1.51 FPS** | Stock VAE enc `193.0 ms` + 2×UNet `97.7 ms` (10.2 FPS) + dec `372.3 ms` |
| **SD-Turbo + Stock AutoencoderKL — 4-Step** | `4` | **771.78 ms** | 783.88 ms | **1.30 FPS** | Stock VAE enc `193.0 ms` + 4×UNet `206.5 ms` (4.8 FPS) + dec `372.3 ms` |

- **Hardware Bottleneck Discovery on GB10:** Stock SD `AutoencoderKL` spends **`565.3 ms`** (`93%` of 1-step frame time!) on memory-bandwidth-bound 512×512 full-res conv layers, whereas the UNet itself executes a 1-step pass in **`40.19 ms` (`24.9 FPS`)**. Swapping to the drop-in `9.8 MB` **`madebyollin/taesd`** VAE collapses encode+decode from `565.3 ms` → **`24.9 ms`** (**22.7× faster VAE** / **6.9× faster E2E**, achieving **11.4 FPS** — exceeding the 10 Hz Frostmarsh simulation tick!).
- **VRAM Footprint:** `3,385.1 MB` peak allocated (`3,572.0 MB` peak reserved), coexisting smoothly on the 128 GB GB10 alongside parallel workloads.

### 3. Temporal Flicker Evaluation (8 Live Frostmarsh Battle Frames)

Evaluated across 8 consecutive 10 Hz live Frostmarsh frames with smooth 1.5°/frame camera orbit + live fire/river simulation (`2-step` SD-Turbo, `strength=0.72`). We report optical-flow-compensated temporal error **`E_warp`** (`mean |O_t - warp(O_{t-1}, flow)|`, isolating pure diffusion hallucination flicker from true camera motion):

| Rendering Mode (8-Frame Live Sequence) | Unwarped L1 | Warped Flicker `E_warp` L1 (0..1) | `E_warp` L1 (0..255) | `E_warp` RMSE | Temporal PSNR | Flicker Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 3 3DGS Input Baseline** *(reference)* | `0.06222` | `0.00808` | `2.06` | `0.01745` | **35.18 dB** | *(geometric baseline)* |
| **Raw img2img (Unseeded per frame)** | `0.12790` | `0.12155` | `30.99` | `0.17434` | **15.18 dB** | — |
| **Raw img2img (Fixed seed, no flow warp)** | `0.04854` | `0.04311` | `10.99` | `0.07304` | **22.78 dB** | `-64.5%` vs raw |
| **Stabilized (Optical-Flow Noise Warp + Motion-Gated Latent EMA)** | `0.04160` | **`0.02911`** | **`7.42`** | **`0.04470`** | **27.04 dB** | **`-76.1%` vs raw / `-32.5%` vs fixed-seed** |

- **Stabilization Mechanism & Overhead:** Dense 3-level GPU Lucas-Kanade optical flow (`7.60 ms` median; in a native 3DGS engine, exact analytical camera motion vectors take `<0.1 ms`!) warps both the `64×64` Gaussian input noise (variance-normalized with `6%` fresh injection) and the previous denoised latent, blended with an occlusion/motion-gated EMA weight (`w ∈ [0.18, 0.64]`, overhead **`2.57 ms`**). This boosts temporal PSNR by **+11.86 dB** over raw img2img while preserving leaping wildfire motion.

### 4. Visual Showcase — 3 Swappable AAA Client Styles from Exact Same Server Tick
![Tier 4 Multi-Style Showcase](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tier4_multistyle.png)

---

## Output File Inventory

- [`tier3_3dgs.py`](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tier3_3dgs.py) — End-to-end Tier 3 State-to-3DGS generator, Triton differentiable tile rasterizer, ONNX exporter & decoupled benchmark suite
- [`tier3_gs_gen.onnx`](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tier3_gs_gen.onnx) — Self-contained ONNX opset 17 3D Gaussian generator (`369.7 KB`, `93,860` params, verified in ORT)
- [`tier3_gs_gen.pt`](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tier3_gs_gen.pt) — PyTorch checkpoint for `GaussianGeneratorNet`
- [`tier3_3dgs_orbit.png`](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tier3_3dgs_orbit.png) — 4-view 3D orbit render showcase of the live Frostmarsh 3D Gaussian cloud
- [`tier4_diffusion.py`](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tier4_diffusion.py) — Tier 4 SD-Turbo multi-style renderer, 1/2/4-step (Stock VAE + TAESD) benchmarks, and optical-flow/latent-EMA temporal stabilizer
- [`tier4_multistyle.png`](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tier4_multistyle.png) — 4-column showcase (Raw 3D `A+` Projection | Dark Fantasy Photoreal | Studio Ghibli Painterly | Cyberpunk Neon Tactical)
- [`tiers_3_4_metrics.json`](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tiers_3_4_metrics.json) — Complete structured JSON metrics for both Tier 3 and Tier 4 (also saved individually as [`tier3_metrics.json`](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tier3_metrics.json) and [`tier4_metrics.json`](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/tier4_metrics.json))
- [`TIERS_3_4_REPORT.md`](/home/gooning/.t3/scratch/frostmarsh-tiers-3-4/TIERS_3_4_REPORT.md) — This report
