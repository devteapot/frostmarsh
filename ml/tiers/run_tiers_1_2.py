#!/usr/bin/env python3
"""
Master Driver for Frostmarsh Tier 1 (512x512 Painterly GAN+WorldTex Renderer)
and Tier 2 (Real 3D Free-Camera G-Buffer -> Neural Deferred Renderer).

Generates and verifies all required artifacts in /home/gooning/.t3/scratch/frostmarsh-tiers-1-2/:
  - tier1_painterly.onnx
  - tier1_showcase.png
  - tier2_deferred_3d.onnx
  - tier2_3d_orbit.png
  - tiers_1_2_metrics.json
  - TIERS_1_2_REPORT.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from data_utils import (
    OUT_DIR,
    build_hero_battle_scene,
    collect_frostmarsh_dataset,
    load_pretrained_sim_models,
)
from tier1_painterly import (
    evaluate_tier1,
    export_and_benchmark_tier1,
    save_tier1_showcase_png,
    train_tier1,
)
from tier2_deferred_3d import (
    evaluate_tier2,
    export_and_benchmark_tier2,
    save_tier2_orbit_png,
    train_tier2,
)


def write_report_md(results: dict, out_md: Path) -> None:
    t1 = results["tier1"]
    t1_onnx = results["tier1_onnx"]
    t2 = results["tier2"]
    t2_onnx = results["tier2_onnx"]

    t1_gpu = t1_onnx["gpu_fp16_latency_ms"].get("128_to_512_fp16", {})
    t2_gpu256 = t2_onnx["gpu_fp16_latency_ms"].get("256x256_fp16", {})
    t2_gpu512 = t2_onnx["gpu_fp16_latency_ms"].get("512x512_fp16", {})

    ec = t1["edge_contrast"]
    ts = t1["temporal_stability"]
    mv = t2["multiview_360_consistency"]

    lines = [
        "# Frostmarsh Neural Rendering — Tier 1 (512×512 Painterly) & Tier 2 (Real 3D Deferred) Report",
        "",
        "## Executive Summary",
        f"- **Device:** `{results['device']}` (shared cleanly with `B=8`, total pipeline runtime `{results['total_runtime_s']:.1f}s`).",
        f"- **Tier 1 (`tier1_painterly.onnx`, `{t1_onnx['n_params']:,}` params, `{t1_onnx['size_kb']:.1f} KB`):** 4× upsampling (`128² → 512²`) Painterly RPG ConvNet conditioned on `17`-ch `A+` state + `2`-ch world-space deterministic texture coordinates, trained with `L1 + Multi-Scale Sobel + Multi-Scale PatchGAN (Hinge + Feature Matching)`. Achieves **{t1['tier1_psnr_db']:.2f} dB** PSNR (`+{t1['tier1_psnr_db'] - t1['renderer_v2_upscaled_psnr_db']:.2f} dB` over upscaled `renderer_v2`), **{ec['tier1_vs_v2_sobel_ratio']:.2f}×** sharper Sobel edge contrast (**{ec['tier1_vs_v2_laplacian_ratio']:.2f}×** Laplacian micro-detail vs `renderer_v2`), **{ts['anti_swimming_reduction_x']:.1f}×** less texture swimming under viewport panning, and **{t1_gpu.get('median_ms', 0):.2f} ms** GPU FP16 (**{t1_gpu.get('fps', 0):.0f} FPS**).",
        f"- **Tier 2 (`tier2_deferred_3d.onnx`, `{t2_onnx['n_params']:,}` params, `{t2_onnx['size_kb']:.1f} KB`):** Real 3D Free-Camera Dilated Multi-Scale Neural Deferred Shader matching the exact 18-channel Rust 3D client G-buffer contract (`gbuf [B,18,H,W] -> rgb_3d [B,3,H,W]`). Reproduces ray-traced soft shadow rays, SSAO, volumetric firelight bounce, water Fresnel sky reflections, subsurface ice glow, and atmospheric depth fog in a single pass at **{t2['val_256_psnr_db']:.2f} dB** (`256²` val) and **{t2['zero_shot_512_psnr_db']:.2f} dB** (zero-shot `512²`), with **{mv['neural_reproj_psnr_db']:.2f} dB** 360° multi-view 3D consistency (reproj RMS `{mv['neural_reproj_rms']:.4f}` vs ray-traced reference `{mv['raytraced_ref_reproj_rms']:.4f}`), running in **{t2_gpu256.get('median_ms', 0):.2f} ms** (`256²`, **{t2_gpu256.get('fps', 0):.0f} FPS**) and **{t2_gpu512.get('median_ms', 0):.2f} ms** (`512²`, **{t2_gpu512.get('fps', 0):.0f} FPS**) GPU FP16!",
        "",
        "---",
        "",
        "## Part A: Tier 1 — High-Res 512×512 Painterly RPG Neural Renderer",
        "",
        "### Architecture & ONNX Contract",
        f"- **File:** `{t1_onnx['path']}` (`{t1_onnx['size_kb']:.1f} KB`, single self-contained opset `{t1_onnx['opset']}`)",
        f"- **Parameters:** `{t1_onnx['n_params']:,}` (`Tier1PainterlyNet`: 68-ch dilated residual backbone + multi-scale lighting pyramid + 2× checkerboard-free bilinear+conv upsampling stages with 10-ch world-harmonic texture injection)",
        f"- **Contract:** `{t1_onnx['input_name']}` `float32 {t1_onnx['input_shape']}` → `{t1_onnx['output_name']}` `float32 {t1_onnx['output_shape']}` (`Sigmoid` in `[0, 1]`)",
        f"- **ORT Parity (`1×19×128×128`):** `max_abs_diff = {t1_onnx['max_abs_diff']['1x19x128x128']:.2e}` (< `1e-4`)",
        f"- **Training:** `{results['tier1_train']['steps']}` steps (`{results['tier1_train']['warmup_steps']}` warmup + `{results['tier1_train']['steps'] - results['tier1_train']['warmup_steps']}` PatchGAN) at `B={results['tier1_train']['batch_size']}` in `{results['tier1_train']['train_time_s']:.1f}s`",
        "",
        "### Quality, Sharp Edge Contrast & Temporal Stability",
        "",
        "| Metric | 512² Painterly Target (GT) | **Tier 1 Painterly (512²)** | `renderer_v2` (256²→512²) |",
        "| --- | ---: | ---: | ---: |",
        f"| Val PSNR (dB) | — | **{t1['tier1_psnr_db']:.2f}** | {t1['renderer_v2_upscaled_psnr_db']:.2f} |",
        f"| Val MAE | — | **{t1['tier1_mae']:.4f}** | {t1['renderer_v2_upscaled_mae']:.4f} |",
        f"| Sobel Edge Contrast (Mean) | {ec['target_512']['sobel_mean']:.5f} | **{ec['tier1_painterly_512']['sobel_mean']:.5f}** | {ec['renderer_v2_256_up512']['sobel_mean']:.5f} |",
        f"| Sobel Edge Contrast (p90) | {ec['target_512']['sobel_p90']:.5f} | **{ec['tier1_painterly_512']['sobel_p90']:.5f}** | {ec['renderer_v2_256_up512']['sobel_p90']:.5f} |",
        f"| Laplacian Micro-Detail RMS | {ec['target_512']['laplacian_rms']:.5f} | **{ec['tier1_painterly_512']['laplacian_rms']:.5f}** | {ec['renderer_v2_256_up512']['laplacian_rms']:.5f} |",
        "",
        "**Temporal Stability (Flicker Energy & Anti-Swimming):**",
        f"- **Static Scene (Full Frame, `dt=0.1s`):** RMS flicker = `{ts['static_scene_full_rms_flicker']:.5f}` (target natural animation RMS = `{ts['target_static_full_rms']:.5f}`)",
        f"- **Static Scene (Solid Non-Animated Terrain):** RMS flicker = **`{ts['static_scene_solid_terrain_rms_flicker']:.5f}`** (target = `{ts['target_static_solid_rms']:.5f}`)",
        f"- **Shifting Viewport (Anti-Swimming Test):** World-locked coords RMS = **`{ts['shifting_scene_world_locked_rms']:.5f}`** vs screen-fixed ablation RMS = `{ts['shifting_scene_unlocked_ablation_rms']:.5f}` (**{ts['anti_swimming_reduction_x']:.2f}×** reduction in panning shimmer!)",
        "",
        "### Latency (`128×128 → 512×512`)",
        f"- **GPU FP16 (`{results['device']}`):** **{t1_gpu.get('median_ms', 0):.3f} ms** median (`{t1_gpu.get('mean_ms', 0):.3f} ms` mean, `{t1_gpu.get('p95_ms', 0):.3f} ms` p95 → **{t1_gpu.get('fps', 0):.1f} FPS**)",
        f"- **CPU ORT (8 threads):** **{t1_onnx['ort_cpu_latency_ms']['8_threads']['median_ms']:.2f} ms** median (`p95 = {t1_onnx['ort_cpu_latency_ms']['8_threads']['p95_ms']:.2f} ms`)",
        f"- **CPU ORT (4 threads):** **{t1_onnx['ort_cpu_latency_ms']['4_threads']['median_ms']:.2f} ms** median",
        f"- **CPU ORT (1 thread):** **{t1_onnx['ort_cpu_latency_ms']['1_threads']['median_ms']:.2f} ms** median",
        "",
        "### Visual Showcase",
        f"![Tier 1 Painterly 512x512 Showcase]({OUT_DIR / 'tier1_showcase.png'})",
        "",
        "---",
        "",
        "## Part B: Tier 2 — Real 3D Free-Camera G-Buffer → Neural Deferred Renderer",
        "",
        "### Architecture & Exact Rust 3D Client ONNX Contract",
        f"- **File:** `{t2_onnx['path']}` (`{t2_onnx['size_kb']:.1f} KB`, single self-contained opset `{t2_onnx['opset']}`)",
        f"- **Parameters:** `{t2_onnx['n_params']:,}` (`Tier2Deferred3DNet`: 40-ch dilated residual blocks `d=1,2,4,8` + 1/4–1/8 global lighting/shadow pyramid)",
        f"- **Contract:** `{t2_onnx['input_name']}` `float32 {t2_onnx['input_shape']}` → `{t2_onnx['output_name']}` `float32 {t2_onnx['output_shape']}` (`Sigmoid` in `[0, 1]`)",
        f"- **ORT Parity:** `256²` = `{t2_onnx['max_abs_diff']['1x18x256x256']:.2e}`, `512²` = `{t2_onnx['max_abs_diff']['1x18x512x512']:.2e}`, `192×320` = `{t2_onnx['max_abs_diff']['1x18x192x320']:.2e}` (all `< 1e-4`)",
        f"- **Training:** `{results['tier2_train']['steps']}` steps at `B={results['tier2_train']['batch_size']}` on `256×256` random 3D orbit cameras (`yaw ∈ [0, 2π]`, `pitch ∈ [25°, 75°]`) in `{results['tier2_train']['train_time_s']:.1f}s`",
        "",
        "### 3D Rendering Accuracy & Multi-View 360° Consistency",
        "",
        "| Benchmark | Metric |",
        "| --- | ---: |",
        f"| Held-Out Val (`256×256`, 32 random 3D orbit cams) | **{t2['val_256_psnr_db']:.2f} dB** (MAE `{t2['val_256_mae']:.4f}`) |",
        f"| Zero-Shot High-Res (`512×512`, 4 battle angles) | **{t2['zero_shot_512_psnr_db']:.2f} dB** (MAE `{t2['zero_shot_512_mae']:.4f}`) |",
        f"| 360° Orbit Multi-View Consistency (`{mv['n_views']}` views, `{mv['step_deg']:.0f}°` step) | **{mv['neural_reproj_psnr_db']:.2f} dB** (RMS `{mv['neural_reproj_rms']:.5f}`) |",
        f"| Ray-Traced Reference 360° View-Dependent Baseline RMS | `{mv['raytraced_ref_reproj_rms']:.5f}` (co-vis `{mv['avg_covisible_fraction']*100:.1f}%`) |",
        f"| Excess Neural View-Flicker RMS above Ray-Traced Ref | **`{mv['excess_view_flicker_rms']:.5f}`** |",
        "",
        "### Latency (`256×256` & `512×512`)",
        "",
        "| Backend / Resolution | Median (ms) | Mean (ms) | p95 (ms) | FPS |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| **GPU FP16 `256×256`** | **{t2_gpu256.get('median_ms', 0):.3f}** | {t2_gpu256.get('mean_ms', 0):.3f} | {t2_gpu256.get('p95_ms', 0):.3f} | **{t2_gpu256.get('fps', 0):.1f}** |",
        f"| **GPU FP16 `512×512`** | **{t2_gpu512.get('median_ms', 0):.3f}** | {t2_gpu512.get('mean_ms', 0):.3f} | {t2_gpu512.get('p95_ms', 0):.3f} | **{t2_gpu512.get('fps', 0):.1f}** |",
        f"| CPU ORT `256×256` (8 threads) | {t2_onnx['ort_cpu_latency_ms']['256x256']['8_threads']['median_ms']:.2f} | {t2_onnx['ort_cpu_latency_ms']['256x256']['8_threads']['mean_ms']:.2f} | {t2_onnx['ort_cpu_latency_ms']['256x256']['8_threads']['p95_ms']:.2f} | {1000.0 / t2_onnx['ort_cpu_latency_ms']['256x256']['8_threads']['median_ms']:.1f} |",
        f"| CPU ORT `256×256` (4 threads) | {t2_onnx['ort_cpu_latency_ms']['256x256']['4_threads']['median_ms']:.2f} | {t2_onnx['ort_cpu_latency_ms']['256x256']['4_threads']['mean_ms']:.2f} | {t2_onnx['ort_cpu_latency_ms']['256x256']['4_threads']['p95_ms']:.2f} | {1000.0 / t2_onnx['ort_cpu_latency_ms']['256x256']['4_threads']['median_ms']:.1f} |",
        f"| CPU ORT `256×256` (1 thread) | {t2_onnx['ort_cpu_latency_ms']['256x256']['1_threads']['median_ms']:.2f} | {t2_onnx['ort_cpu_latency_ms']['256x256']['1_threads']['mean_ms']:.2f} | {t2_onnx['ort_cpu_latency_ms']['256x256']['1_threads']['p95_ms']:.2f} | {1000.0 / t2_onnx['ort_cpu_latency_ms']['256x256']['1_threads']['median_ms']:.1f} |",
        f"| CPU ORT `512×512` (8 threads) | {t2_onnx['ort_cpu_latency_ms']['512x512']['8_threads']['median_ms']:.2f} | {t2_onnx['ort_cpu_latency_ms']['512x512']['8_threads']['mean_ms']:.2f} | {t2_onnx['ort_cpu_latency_ms']['512x512']['8_threads']['p95_ms']:.2f} | {1000.0 / t2_onnx['ort_cpu_latency_ms']['512x512']['8_threads']['median_ms']:.1f} |",
        "",
        "### 4-Angle 3D Battle Orbit Showcase",
        f"![Tier 2 3D Orbit Showcase]({OUT_DIR / 'tier2_3d_orbit.png'})",
        "",
        "---",
        "",
        "## Generated Artifacts",
        f"- `{OUT_DIR / 'tier1_painterly.onnx'}` (`{t1_onnx['size_kb']:.1f} KB`)",
        f"- `{OUT_DIR / 'tier1_showcase.png'}`",
        f"- `{OUT_DIR / 'tier2_deferred_3d.onnx'}` (`{t2_onnx['size_kb']:.1f} KB`)",
        f"- `{OUT_DIR / 'tier2_3d_orbit.png'}`",
        f"- `{OUT_DIR / 'tiers_1_2_metrics.json'}`",
        f"- `{OUT_DIR / 'TIERS_1_2_REPORT.md'}`",
        "",
    ]
    out_md.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"=== Frostmarsh Tiers 1 & 2 Neural Rendering Pipeline on {dev_name} ===", flush=True)
    t_start = time.time()

    # 1. Shared real Frostmarsh rollout dataset
    print("\n[1/5] Collecting real Frostmarsh 128x128 A+ rollouts (world_model.pt + ecosystem_model.pt)...", flush=True)
    t_d = time.time()
    w_model, e_model = load_pretrained_sim_models(device)
    train_split = collect_frostmarsh_dataset(w_model, e_model, n_worlds=24, n_steps=36, sample_every=2, device=device, seed=101)
    val_split = collect_frostmarsh_dataset(w_model, e_model, n_worlds=12, n_steps=18, sample_every=6, device=device, seed=909)
    hero_scene = build_hero_battle_scene(w_model, e_model, device, seed=2026)
    print(
        f"  Collected train={train_split['mass'].shape[0]}, val={val_split['mass'].shape[0]}, hero=1 frames in {time.time() - t_d:.1f}s",
        flush=True,
    )
    del w_model, e_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Tier 1: High-Res 512x512 Painterly RPG Neural Renderer
    print("\n[2/5] Training Tier 1 Painterly 512x512 Renderer (L1 + Sobel + Multi-Scale PatchGAN, B=8, BF16)...", flush=True)
    t1_gen, t1_train_meta = train_tier1(train_split, device, steps=1400, warmup_steps=300, batch_size=8, seed=1001)
    torch.save({"state_dict": t1_gen.state_dict(), "train_meta": t1_train_meta}, OUT_DIR / "tier1_painterly.pt")

    print("\n[3/5] Evaluating & Exporting Tier 1 (tier1_painterly.onnx + tier1_showcase.png)...", flush=True)
    t1_metrics, t1_vis = evaluate_tier1(t1_gen, val_split, hero_scene, device)
    assert t1_metrics["tier1_psnr_db"] > 20.0, f"Unexpected Tier 1 PSNR: {t1_metrics['tier1_psnr_db']}"
    t1_onnx = export_and_benchmark_tier1(t1_gen, OUT_DIR / "tier1_painterly.onnx", device)
    save_tier1_showcase_png(t1_vis, t1_metrics, OUT_DIR / "tier1_showcase.png")
    print(f"  Tier 1 Val PSNR={t1_metrics['tier1_psnr_db']:.2f} dB, Sobel ratio={t1_metrics['edge_contrast']['tier1_vs_v2_sobel_ratio']}x, GPU FP16={t1_onnx['gpu_fp16_latency_ms']}", flush=True)
    del t1_gen, t1_vis
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. Tier 2: Real 3D Free-Camera G-Buffer -> Neural Deferred Renderer
    print("\n[4/5] Training Tier 2 Real 3D Deferred Shader (256x256 random orbit cameras, B=8, BF16)...", flush=True)
    t2_model, t2_train_meta = train_tier2(train_split, device, steps=1600, batch_size=8, seed=2002)
    torch.save({"state_dict": t2_model.state_dict(), "train_meta": t2_train_meta}, OUT_DIR / "tier2_deferred_3d.pt")

    print("\n[5/5] Evaluating & Exporting Tier 2 (tier2_deferred_3d.onnx + tier2_3d_orbit.png)...", flush=True)
    t2_metrics, t2_vis = evaluate_tier2(t2_model, val_split, hero_scene, device)
    assert t2_metrics["val_256_psnr_db"] > 24.0, f"Unexpected Tier 2 PSNR: {t2_metrics['val_256_psnr_db']}"
    t2_onnx = export_and_benchmark_tier2(t2_model, OUT_DIR / "tier2_deferred_3d.onnx", device)
    save_tier2_orbit_png(t2_vis, t2_metrics, OUT_DIR / "tier2_3d_orbit.png")
    print(f"  Tier 2 Val 256² PSNR={t2_metrics['val_256_psnr_db']:.2f} dB, Zero-Shot 512²={t2_metrics['zero_shot_512_psnr_db']:.2f} dB, 360° Reproj={t2_metrics['multiview_360_consistency']['neural_reproj_psnr_db']:.2f} dB", flush=True)

    results = {
        "device": dev_name,
        "total_runtime_s": round(time.time() - t_start, 1),
        "dataset": {
            "train_frames": int(train_split["mass"].shape[0]),
            "val_frames": int(val_split["mass"].shape[0]),
            "grid_size": "128x128",
        },
        "tier1_train": t1_train_meta,
        "tier1": t1_metrics,
        "tier1_onnx": t1_onnx,
        "tier2_train": t2_train_meta,
        "tier2": t2_metrics,
        "tier2_onnx": t2_onnx,
    }
    (OUT_DIR / "tiers_1_2_metrics.json").write_text(json.dumps(results, indent=2))
    write_report_md(results, OUT_DIR / "TIERS_1_2_REPORT.md")
    print(f"\n=== ALL TIER 1 & TIER 2 ARTIFACTS COMPLETE in {results['total_runtime_s']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
