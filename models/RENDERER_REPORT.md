# Frostmarsh Client-Side Neural State-to-Frame Renderer Report

## Summary
- **ONNX Model:** `/home/gooning/.t3/scratch/frostmarsh-renderer/renderer_model.onnx` (`402.2 KB` / `411,862` bytes, single self-contained file, opset 17)
- **Parameters:** `99,747` (`base_ch=32`, `up_ch=16`, `glow_ch=24`)
- **Validation PSNR (60 held-out scenes via ONNXRuntime):** **`35.14 dB`** (MAE `0.0078`)
- **CPU Inference Latency (`onnxruntime` 1.32.0, 128×128 → 256×256):**
  - 8 threads: **`6.69 ms`** mean (`6.63 ms` median, `6.86 ms` p95)
  - 4 threads: **`10.15 ms`** mean (`10.15 ms` median)
  - 1 thread: **`34.90 ms`** mean (`34.88 ms` median)
- **GPU Inference Latency (`NVIDIA GB10`):**
  - PyTorch CUDA Graph FP32: **`0.633 ms`**
  - PyTorch CUDA Eager FP16: **`0.558 ms`**
  - PyTorch CUDA Eager FP32: **`0.802 ms`**
  - ONNXRuntime available providers on host: `['AzureExecutionProvider', 'CPUExecutionProvider']`
- **Sample Comparison Grid:** `/home/gooning/.t3/scratch/frostmarsh-renderer/sample_render.png`

## ONNX Contract Verification (`crates/neural-client/src/main.rs`)
- **Input tensor:** `frame_state` — `tensor(float)`, shape `['batch', 20, 'height', 'width']` (dynamic `B, H, W`)
- **Output tensor:** `rgb_frame` — `tensor(float)`, shape `['batch', 3, 'out_height', 'out_width']`, strictly in `[0.0, 1.0]` (`Sigmoid`)
- **PyTorch vs ONNXRuntime max abs diff:**
  - `1×20×128×128 → 1×3×256×256`: `4.83e-06`
  - `2×20×128×128 → 2×3×256×256`: `5.54e-06`
  - `1×20×64×64 → 1×3×128×128`: `3.81e-06`

## Architecture
- **Stem:** `Conv3x3(20 → 32) → GELU`
- **Trunk:** 3× `DilatedResBlock(32)` with dilations `(1, 2, 4)` + de-gridding `3×3` second conv
- **Wide-Context Lighting Branch:** `Conv1x1(32 → 24) → AvgPool(4) → Conv3x3 → [AvgPool(2) → Conv3x3 → BilinearUp] → Conv3x3 → BilinearUp → Conv1x1(24 → 32)` (adds smooth 48px-radius dynamic lighting for fire/frost/relic halos at <2% FLOPs)
- **2× Upsampling Head:** `Conv3x3(32 → 64) → PixelShuffle(2) → Conv3x3(16 → 16) → GELU → Conv3x3(16 → 3) → Sigmoid`

## Training Configuration & Per-Archetype Accuracy
- **Loss:** `L1 + 0.2 * Sobel_edge_loss`
- **Schedule:** `5,000` steps × batch `16` (`80,000` online synthetic frames), AdamW (`lr=0.003` cosine decay), `589.7s` on GPU.

| Scene Archetype | Validation PSNR |
| --- | ---: |
| `river_skirmish` | 36.50 dB |
| `forest_wildfire` | 34.46 dB |
| `frozen_crossing` | 33.85 dB |
| `oil_marsh_inferno` | 33.68 dB |
| `highland_shrine` | 37.87 dB |
| `combat_showcase` | 36.01 dB |

## Sample Render Grid (`sample_render.png`)
| Showcase Scene | ONNX PSNR | MAE |
| --- | ---: | ---: |
| `river_skirmish` | 35.79 dB | 0.0066 |
| `forest_wildfire` | 33.24 dB | 0.0069 |
| `frozen_crossing` | 33.71 dB | 0.0125 |
| `oil_marsh_inferno` | 34.50 dB | 0.0096 |
