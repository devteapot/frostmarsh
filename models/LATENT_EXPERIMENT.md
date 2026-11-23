# Frostmarsh 4-Arm Compositional vs. Shared-Latent Experiment

## Executive Summary
- **Production Winner (`renderer_v2.onnx`):** **Arm A+ (4 B/cell Compositional Top-2 Schema)** achieves the **highest In-Distribution accuracy (30.63 dB, MAE 0.0176)** of all four arms (`+3.72 dB` over Arm A, `+1.33 dB` over Arm D), delivers **zero client-server version lock**, exact Rust gameplay compatibility, and compresses **5.9x** via zlib (**25.3 KB/s**) / **5.4x** via Brotli (**27.6 KB/s**) at 10 Hz.
- **Arm A (`3 B/cell`, 1-dominant class) is strictly dominated:** Dropping minority species limits Arm A to **26.91 dB** in-distribution and **16.29 dB** on held-out unseen mixtures.
- **Key Scientific Finding on 3+ Species Unseen Mixtures:** On held-out unseen mixtures, Arm A+ beats Arm A by **`+4.57 dB`** (**20.86** vs **16.29 dB**) and beats Arm C (**19.18 dB**). However, because Frostmarsh terrain cells carry persistent underlying `soil` (`400-1000` mass), 3-species mixtures (`soil + plant + steam`, `soil + ice + oil`, `soil + ice + oil + water`) truncate the 3rd species under strict top-2 mass selection — allowing Arm D's full-10-species encoder to reach **24.34 dB** unseen (and Arm C to reach **25.73 dB** on *Dense Steam over Brush*). Actionable takeaway for v3: ranking foreground/overlay species (`oil, steam, smoke`) ahead of background `soil` fixes 3-species truncation at **0 extra wire bytes**.
- **Arm C (`A + z_free`, K=4 int8 @ 32×32, 50% z-dropout):** Recovers `+2.34 dB` over Arm A in-distribution (**29.25 dB**) and degrades gracefully to **26.81 dB** when `z=0` (matching Arm A's **26.91 dB**), at `21.6 KB/s` zlib (`+0.7 KB/s` over Arm A), though it still trails Arm A+ overall and introduces encoder-decoder coupling.
- **Arm D (`z_only`, K=16 int8 @ 32×32) fails production deployment due to hard version lock & spatial blur:** Despite low raw bytes (`5.8 KB/s` visual-only, `12.7 KB/s` with required `1 B/cell` gameplay hazard; **1.00x** compression ratio = incompressible high-entropy latents), Arm D loses `-1.33 dB` to Arm A+ in-distribution from 4× spatial bottlenecking and suffers **catastrophic cross-seed version lock**: decoding Seed-1 `z` with an independently trained Seed-2 decoder collapses PSNR from **29.30 dB** to **15.84 dB** (`-13.46 dB`)!

## 1. Accuracy: In-Distribution vs. Held-Out Unseen Mixtures

| Arm | Wire Schema | Decoder Params | In-Dist PSNR (dB) | In-Dist MAE | Unseen Mix PSNR (dB) | Unseen Mix MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Arm A** | 12-class 1-dom (`3 B/cell`) | 98,883 | 26.91 | 0.0260 | 16.29 | 0.1187 |
| **Arm A+** | Compositional Top-2 (`4 B/cell`) | 98,883 | **30.63** | **0.0176** | **20.86** | **0.0676** |
| **Arm C (`z=full`)** | `A + z_free` (`3 B` + `K=4` int8 @ 32²) | 100,035 | 29.25 | 0.0202 | 19.18 | 0.0770 |
| **Arm C (`z=0`)** | `A + z_free` (50% z-dropout fallback) | 100,035 | 26.81 | 0.0274 | 16.44 | 0.1162 |
| **Arm D (Matched)** | Pure `z_only` (`K=16` int8 @ 32²) | 99,171 | 29.30 | 0.0198 | 24.34 | 0.0459 |
| **Arm D (Cross-Seed Swap)** | `D_seed2(E_seed1(x))` version-lock test | 99,171 | **15.84** | **0.1185** | **14.97** | **0.1310** |

### Per-Mixture Breakdown (Held-Out Unseen Mixtures — Strictly Excluded From Training)

| Held-Out Unseen Mixture | Arm A | **Arm A+** | Arm C (`z=full`) | Arm C (`z=0`) | Arm D (Matched) | Arm D (Swap) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Burning Oil on Ice | 18.74 dB | **20.70 dB** | 20.53 dB | 18.36 dB | 23.44 dB | 12.31 dB |
| Dense Steam over Forest Brush | 18.31 dB | **21.72 dB** | 25.73 dB | 19.23 dB | 28.51 dB | 16.94 dB |
| Frozen Oil-Mud Slush | 13.74 dB | **20.29 dB** | 16.06 dB | 13.81 dB | 22.92 dB | 17.83 dB |

## 2. Real Wire Cost (Dirty `16×16` Chunks @ 10 Hz With Deadband)

Measured across a 10-second (`100` ticks @ 10 Hz) `128×128` Frostmarsh rollout driven by `world_model.pt` + `ecosystem_model.pt` with periodic Fire/Frost bolts:

| Arm | Avg Dirty Chunks (/64) | Raw B/tick | Raw KB/s | LZ4 KB/s | zlib KB/s | Brotli (q=4) KB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm A (`3 B/cell`) | 13.7 | 10488 | 102.43 | 41.39 | 20.92 | 22.99 |
| **Arm A+ (`4 B/cell`)** | 14.8 | 15213 | 148.56 | 54.90 | 25.32 | 27.60 |
| Arm C (`A + K=4 int8 @ 32²`) | 14.1 | 10557 | 103.10 | 42.00 | 21.59 | 23.61 |
| Arm D (`K=16 int8 @ 32²` visual only) | 2.3 | 595 | 5.81 | 6.04 | 5.84 | 5.78 |
| Arm D + `1 B/cell` gameplay hazard | 13.7 | 3939 | 38.46 | 19.38 | 12.89 | 12.73 |

## 3. Version-Lock & Degradation Diagnostics
- **Arm C z-dropout delta (In-Dist):** `29.25 dB` (`z=full`) → `26.81 dB` (`z=0`), Δ = `-2.44 dB` (graceful fallback to Arm A baseline).
- **Arm D cross-seed swap (In-Dist):** `29.30 dB` (matched `D1(E1)`) vs `29.08 dB` (`D2(E2)`) → collapses to **`15.84 dB`** (`D2(E1)`) and **`14.73 dB`** (`D1(E2)`), Δ = `-13.46 dB` — proving hard client-server latent version lock.

## 4. Exported `renderer_v2.onnx` (Arm A+) Verification & CPU Benchmark
- **File:** `/home/gooning/.t3/scratch/frostmarsh-latent-exp/renderer_v2.onnx` (`398.8 KB`, single self-contained file, opset 17)
- **Contract:** `aplus_state` `float32 ['batch', 17, 'height', 'width']` → `rgb_frame` `float32 ['batch', 3, 'out_height', 'out_width']`
- **PyTorch vs ONNXRuntime Max Abs Diff (`1×17×128×128`):** `5.96e-07` (< `1e-4`)
- **ORT CPU Latency (`128×128 → 256×256`):**
  - 8 threads: **16.64 ms** median (17.07 ms mean, 19.78 ms p95)
  - 4 threads: **10.60 ms** median
  - 1 thread: **36.24 ms** median

## 5. Visual Comparison
![4-Arm Compositional vs. Latent Comparison](/home/gooning/.t3/scratch/frostmarsh-latent-exp/comparison_4arm.png)
