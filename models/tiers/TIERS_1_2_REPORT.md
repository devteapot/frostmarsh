# Frostmarsh Neural Rendering — Tier 1 (512×512 Painterly) & Tier 2 (Real 3D Deferred) Report

## Executive Summary
- **Device:** `NVIDIA GB10` (shared cleanly with `B=8`, total pipeline runtime `736.1s`).
- **Tier 1 (`tier1_painterly.onnx`, `387,719` params, `1540.0 KB`):** 4× upsampling (`128² → 512²`) Painterly RPG ConvNet conditioned on `17`-ch `A+` state + `2`-ch world-space deterministic texture coordinates, trained with `L1 + Multi-Scale Sobel + Multi-Scale PatchGAN (Hinge + Feature Matching)`. Achieves **22.86 dB** PSNR (`+3.16 dB` over upscaled `renderer_v2`), **2.00×** sharper Sobel edge contrast (**1.71×** Laplacian micro-detail vs `renderer_v2`), **24.4×** less texture swimming under viewport panning, and **4.30 ms** GPU FP16 (**233 FPS**).
- **Tier 2 (`tier2_deferred_3d.onnx`, `162,211` params, `647.7 KB`):** Real 3D Free-Camera Dilated Multi-Scale Neural Deferred Shader matching the exact 18-channel Rust 3D client G-buffer contract (`gbuf [B,18,H,W] -> rgb_3d [B,3,H,W]`). Reproduces ray-traced soft shadow rays, SSAO, volumetric firelight bounce, water Fresnel sky reflections, subsurface ice glow, and atmospheric depth fog in a single pass at **27.78 dB** (`256²` val) and **24.56 dB** (zero-shot `512²`), with **34.27 dB** 360° multi-view 3D consistency (reproj RMS `0.0194` vs ray-traced reference `0.0202`), running in **2.09 ms** (`256²`, **478 FPS**) and **11.72 ms** (`512²`, **85 FPS**) GPU FP16!

---

## Part A: Tier 1 — High-Res 512×512 Painterly RPG Neural Renderer

### Architecture & ONNX Contract
- **File:** `/home/gooning/.t3/scratch/frostmarsh-tiers-1-2/tier1_painterly.onnx` (`1540.0 KB`, single self-contained opset `17`)
- **Parameters:** `387,719` (`Tier1PainterlyNet`: 68-ch dilated residual backbone + multi-scale lighting pyramid + 2× checkerboard-free bilinear+conv upsampling stages with 10-ch world-harmonic texture injection)
- **Contract:** `aplus_world` `float32 ['batch', 19, 'height', 'width']` → `rgb_512` `float32 ['batch', 3, 'out_height', 'out_width']` (`Sigmoid` in `[0, 1]`)
- **ORT Parity (`1×19×128×128`):** `max_abs_diff = 1.01e-06` (< `1e-4`)
- **Training:** `1400` steps (`300` warmup + `1100` PatchGAN) at `B=8` in `465.7s`

### Quality, Sharp Edge Contrast & Temporal Stability

| Metric | 512² Painterly Target (GT) | **Tier 1 Painterly (512²)** | `renderer_v2` (256²→512²) |
| --- | ---: | ---: | ---: |
| Val PSNR (dB) | — | **22.86** | 19.70 |
| Val MAE | — | **0.0463** | 0.0657 |
| Sobel Edge Contrast (Mean) | 0.06375 | **0.05246** | 0.02625 |
| Sobel Edge Contrast (p90) | 0.14370 | **0.11987** | 0.05728 |
| Laplacian Micro-Detail RMS | 0.12840 | **0.07027** | 0.04109 |

**Temporal Stability (Flicker Energy & Anti-Swimming):**
- **Static Scene (Full Frame, `dt=0.1s`):** RMS flicker = `0.00367` (target natural animation RMS = `0.01693`)
- **Static Scene (Solid Non-Animated Terrain):** RMS flicker = **`0.00103`** (target = `0.00432`)
- **Shifting Viewport (Anti-Swimming Test):** World-locked coords RMS = **`0.00221`** vs screen-fixed ablation RMS = `0.05379` (**24.38×** reduction in panning shimmer!)

### Latency (`128×128 → 512×512`)
- **GPU FP16 (`NVIDIA GB10`):** **4.296 ms** median (`4.410 ms` mean, `5.143 ms` p95 → **232.8 FPS**)
- **CPU ORT (8 threads):** **51.91 ms** median (`p95 = 52.39 ms`)
- **CPU ORT (4 threads):** **90.80 ms** median
- **CPU ORT (1 thread):** **341.47 ms** median

### Visual Showcase
![Tier 1 Painterly 512x512 Showcase](/home/gooning/.t3/scratch/frostmarsh-tiers-1-2/tier1_showcase.png)

---

## Part B: Tier 2 — Real 3D Free-Camera G-Buffer → Neural Deferred Renderer

### Architecture & Exact Rust 3D Client ONNX Contract
- **File:** `/home/gooning/.t3/scratch/frostmarsh-tiers-1-2/tier2_deferred_3d.onnx` (`647.7 KB`, single self-contained opset `17`)
- **Parameters:** `162,211` (`Tier2Deferred3DNet`: 40-ch dilated residual blocks `d=1,2,4,8` + 1/4–1/8 global lighting/shadow pyramid)
- **Contract:** `gbuf` `float32 ['batch', 18, 'height', 'width']` → `rgb_3d` `float32 ['batch', 3, 'height', 'width']` (`Sigmoid` in `[0, 1]`)
- **ORT Parity:** `256²` = `3.58e-07`, `512²` = `4.17e-07`, `192×320` = `4.17e-07` (all `< 1e-4`)
- **Training:** `1600` steps at `B=8` on `256×256` random 3D orbit cameras (`yaw ∈ [0, 2π]`, `pitch ∈ [25°, 75°]`) in `228.3s`

### 3D Rendering Accuracy & Multi-View 360° Consistency

| Benchmark | Metric |
| --- | ---: |
| Held-Out Val (`256×256`, 32 random 3D orbit cams) | **27.78 dB** (MAE `0.0177`) |
| Zero-Shot High-Res (`512×512`, 4 battle angles) | **24.56 dB** (MAE `0.0267`) |
| 360° Orbit Multi-View Consistency (`36` views, `10°` step) | **34.27 dB** (RMS `0.01941`) |
| Ray-Traced Reference 360° View-Dependent Baseline RMS | `0.02016` (co-vis `63.6%`) |
| Excess Neural View-Flicker RMS above Ray-Traced Ref | **`0.00000`** |

### Latency (`256×256` & `512×512`)

| Backend / Resolution | Median (ms) | Mean (ms) | p95 (ms) | FPS |
| --- | ---: | ---: | ---: | ---: |
| **GPU FP16 `256×256`** | **2.091** | 2.188 | 2.619 | **478.2** |
| **GPU FP16 `512×512`** | **11.721** | 11.870 | 12.856 | **85.3** |
| CPU ORT `256×256` (8 threads) | 31.56 | 31.78 | 32.01 | 31.7 |
| CPU ORT `256×256` (4 threads) | 52.54 | 52.62 | 52.92 | 19.0 |
| CPU ORT `256×256` (1 thread) | 204.26 | 204.24 | 204.96 | 4.9 |
| CPU ORT `512×512` (8 threads) | 117.68 | 117.75 | 118.16 | 8.5 |

### 4-Angle 3D Battle Orbit Showcase
![Tier 2 3D Orbit Showcase](/home/gooning/.t3/scratch/frostmarsh-tiers-1-2/tier2_3d_orbit.png)

---

## Generated Artifacts
- `/home/gooning/.t3/scratch/frostmarsh-tiers-1-2/tier1_painterly.onnx` (`1540.0 KB`)
- `/home/gooning/.t3/scratch/frostmarsh-tiers-1-2/tier1_showcase.png`
- `/home/gooning/.t3/scratch/frostmarsh-tiers-1-2/tier2_deferred_3d.onnx` (`647.7 KB`)
- `/home/gooning/.t3/scratch/frostmarsh-tiers-1-2/tier2_3d_orbit.png`
- `/home/gooning/.t3/scratch/frostmarsh-tiers-1-2/tiers_1_2_metrics.json`
- `/home/gooning/.t3/scratch/frostmarsh-tiers-1-2/TIERS_1_2_REPORT.md`
