"""
Comprehensive Gate Evaluation Suite (G1, G2, G3, G4, G5-B) for Frostmarsh Neural World Model POC.

- G1: Exact physical invariants (mass & total energy conservation, positivity, H_CAP, T bounds,
      wall impermeability, 2nd-law heat conduction) on 128x128 closed-system rollouts.
- G2: 1s (10 ticks), 10s (100 ticks), and 60s (600 ticks) rollout fidelity on held-out Frostmarsh
      scenes at BOTH 128x128 and 256x256 (including Frost Bolt ice-bridge & Fire Bolt wildfire).
- G3: 10,000-step long autoregressive rollout stability & projection-active % telemetry.
- G4: Batched 128x128 & 256x256 GPU latency (p50/p99) + full-step latency + ORT CPU latency.
- G5-B: Aggregates Living Ecosystem Neural Field vs Reaction-Diffusion baseline results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from model import WorldModel, count_parameters
from sim_teacher import (
    H_CAP,
    RefSim,
    S_ASH,
    S_ICE,
    S_ROCK,
    S_WATER,
    T_MAX,
    T_MIN,
    SimState,
    compute_C_and_T,
    compute_total_energy,
    compute_total_mass,
    encode_state,
    make_scene,
    project_and_step,
    sample_bolts,
)

OUT_DIR = Path("/home/gooning/.t3/scratch/frostmarsh-ml")


@torch.no_grad()
def eval_g1_invariants(model: WorldModel, device: torch.device) -> dict:
    """G1: Closed-system invariants over 200 ticks on 128x128 across all 5 Frostmarsh biomes."""
    print("\n=== G1: Physical Invariants (Closed System, 200 ticks, 5x 128x128 scenes) ===")
    state = make_scene(5, 128, 128, device, scene_type="mixed", seed=301)
    m0 = compute_total_mass(state)
    e0 = compute_total_energy(state)

    max_rel_mass_drift = 0.0
    max_rel_energy_drift = 0.0
    min_mass_seen = float("inf")
    max_cell_mass_seen = 0.0
    min_T_seen = float("inf")
    max_T_seen = 0.0
    wall_flux_violations = 0
    second_law_violations = 0

    for t in range(200):
        x = encode_state(state)
        raw = model(x)
        next_state, proj, _ = project_and_step(state, raw)

        # Check wall impermeability on proj_deltas
        wall = (state.mass[:, S_ROCK:S_ROCK + 1] >= 500.0) | (state.mass[:, S_ICE:S_ICE + 1] >= 700.0)
        wall_e = wall | F.pad(wall[:, :, :, 1:], (0, 1, 0, 0), value=True)
        wall_s = wall | F.pad(wall[:, :, 1:, :], (0, 0, 0, 1), value=True)
        if (proj[:, 0:4].abs()[wall_e.expand(-1, 4, -1, -1)] > 1e-7).any():
            wall_flux_violations += 1
        if (proj[:, 4:8].abs()[wall_s.expand(-1, 4, -1, -1)] > 1e-7).any():
            wall_flux_violations += 1

        # Check 2nd-law heat sign
        _, T_cur = compute_C_and_T(state.mass, state.energy)
        dT_e = T_cur[:, :, :, :-1] - T_cur[:, :, :, 1:]
        dT_s = T_cur[:, :, :-1, :] - T_cur[:, :, 1:, :]
        if ((proj[:, 8:9, :, :-1] * dT_e) < -1e-6).any() or ((proj[:, 9:10, :-1, :] * dT_s) < -1e-6).any():
            second_law_violations += 1

        state = next_state

        m_now = compute_total_mass(state)
        e_now = compute_total_energy(state)
        rel_m = float(((m_now - m0).abs() / (m0.abs() + 1e-12)).max().item())
        rel_e = float(((e_now - e0).abs() / (e0.abs() + 1e-12)).max().item())
        max_rel_mass_drift = max(max_rel_mass_drift, rel_m)
        max_rel_energy_drift = max(max_rel_energy_drift, rel_e)

        min_mass_seen = min(min_mass_seen, float(state.mass.min().item()))
        max_cell_mass_seen = max(max_cell_mass_seen, float(state.mass.sum(dim=1).max().item()))
        C_now, T_now = compute_C_and_T(state.mass, state.energy)
        valid_c = C_now > 1e-6
        if valid_c.any():
            min_T_seen = min(min_T_seen, float(T_now[valid_c].min().item()))
            max_T_seen = max(max_T_seen, float(T_now[valid_c].max().item()))

    res = {
        "max_rel_mass_drift": max_rel_mass_drift,
        "max_rel_energy_drift": max_rel_energy_drift,
        "min_mass": min_mass_seen,
        "max_cell_total_mass": max_cell_mass_seen,
        "min_T_cK": min_T_seen,
        "max_T_cK": max_T_seen,
        "wall_flux_violations": wall_flux_violations,
        "second_law_violations": second_law_violations,
        "pass": bool(
            max_rel_mass_drift < 1e-5
            and max_rel_energy_drift < 1e-4
            and min_mass_seen >= 0.0
            and max_cell_mass_seen <= H_CAP + 1e-3
            and min_T_seen >= T_MIN - 1.0
            and max_T_seen <= T_MAX + 1.0
            and wall_flux_violations == 0
            and second_law_violations == 0
        ),
    }
    print(
        f"  mass_drift={max_rel_mass_drift:.2e}, energy_drift={max_rel_energy_drift:.2e}, "
        f"min_m={min_mass_seen:.4f}, max_cell_m={max_cell_mass_seen:.1f}/{H_CAP}, "
        f"T=[{min_T_seen:.0f}, {max_T_seen:.0f}] cK, wall_viol={wall_flux_violations}, "
        f"2ndlaw_viol={second_law_violations} -> PASS={res['pass']}"
    )
    return res


def _iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    inter = (mask_a & mask_b).float().sum().item()
    union = (mask_a | mask_b).float().sum().item()
    return float(inter / union) if union > 0 else 1.0


@torch.no_grad()
def eval_g2_fidelity(model: WorldModel, device: torch.device) -> dict:
    """
    G2: 1s (10 ticks), 10s (100 ticks), and 60s (600 ticks) rollout fidelity
    at both 128x128 and 256x256 across Frostmarsh scenes with scripted Fire & Frost bolts.
    """
    print("\n=== G2: Rollout Fidelity (1s / 10s / 60s) at 128x128 & 256x256 ===")
    results = {}

    for H, W, B in [(128, 128, 5), (256, 256, 2)]:
        gt_state = make_scene(B, H, W, device, scene_type="mixed", seed=500 + H)
        nn_state = gt_state.clone()

        yy = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
        xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)

        # Aim the Frost Bolt directly at the river center in scene 0 (river_crossing) so ice bridge forms reliably!
        river_cx = float(gt_state.mass[0, S_WATER, H // 2].argmax().item())
        river_cy = float(H // 2)
        frost_kernel = torch.clamp(1.0 - ((yy - river_cy) ** 2 + (xx - river_cx) ** 2) / (5.5 ** 2), min=0.0)[0]
        fire_kernel = torch.clamp(1.0 - ((yy - H * 0.45) ** 2 + (xx - W * 0.45) ** 2) / (5.0 ** 2), min=0.0)[0]

        horizons = {10: "1s", 100: "10s", 600: "60s"}
        size_res = {}
        proj_act_acc = 0.0

        for t in range(1, 601):
            fe = torch.zeros((B, 1, H, W), device=device)
            fm = torch.zeros((B, 1, H, W), device=device)

            # Scripted Frost Bolt on river center (ticks 1..8) -> creates walkable ice bridge!
            if 1 <= t <= 8:
                fe[0, 0] -= frost_kernel * 3.5e8
            # Scripted Fire Bolt on forest / marsh (ticks 15..20) -> ignites wildfire!
            if 15 <= t <= 20:
                fe[min(1, B - 1), 0] += fire_kernel * 3.8e8
            # Periodic ambient bolts during 60s rollout
            if t % 80 == 0:
                fe_r, fm_r = sample_bolts(B, H, W, device, prob=0.8, max_bolts=1)
                fe = fe + fe_r
                fm = fm + fm_r

            gt_state, _ = RefSim.step(gt_state, fe, fm)
            raw = model(encode_state(nn_state, fe, fm))
            nn_state, _, active = project_and_step(nn_state, raw, fe, fm)
            proj_act_acc += float(active.float().mean().item())

            if t in horizons:
                h_name = horizons[t]
                mass_mae = float((nn_state.mass - gt_state.mass).abs().mean().item())
                mass_rmse = float(torch.sqrt(((nn_state.mass - gt_state.mass) ** 2).mean()).item())
                _, T_gt = compute_C_and_T(gt_state.mass, gt_state.energy)
                _, T_nn = compute_C_and_T(nn_state.mass, nn_state.energy)
                temp_mae_K = float((T_nn - T_gt).abs().mean().item()) / 100.0

                water_iou = _iou(nn_state.mass[:, S_WATER] > 100.0, gt_state.mass[:, S_WATER] > 100.0)
                ice_iou = _iou(nn_state.mass[:, S_ICE] >= 700.0, gt_state.mass[:, S_ICE] >= 700.0)
                burn_iou = _iou(
                    (nn_state.mass[:, S_ASH] > 50.0) | (T_nn[:, 0] > 44000.0),
                    (gt_state.mass[:, S_ASH] > 50.0) | (T_gt[:, 0] > 44000.0),
                )

                size_res[h_name] = {
                    "mass_mae": mass_mae,
                    "mass_mae_norm": mass_mae / 1000.0,
                    "mass_rmse": mass_rmse,
                    "temp_mae_K": temp_mae_K,
                    "water_iou": water_iou,
                    "ice_bridge_iou": ice_iou,
                    "fire_ash_iou": burn_iou,
                    "proj_active_pct": (proj_act_acc / t) * 100.0,
                }
                print(
                    f"  [{H}x{W} @ {h_name:3s}] mass_MAE={mass_mae:6.2f} ({mass_mae/1000:.4f}) | "
                    f"T_MAE={temp_mae_K:5.2f} K | water_IoU={water_iou:.3f} | "
                    f"ice_IoU={ice_iou:.3f} | fire_IoU={burn_iou:.3f} | proj_act={proj_act_acc/t*100:.2f}%"
                )

        results[f"{H}x{W}"] = size_res
    return results


@torch.no_grad()
def eval_g3_long_rollout(model: WorldModel, device: torch.device, total_steps: int = 10000) -> dict:
    """G3: 10,000-step (1,000s) autoregressive rollout stability & projection-active % telemetry."""
    print(f"\n=== G3: {total_steps:,}-Step Long Rollout Stability (128x128) ===")
    state = make_scene(1, 128, 128, device, scene_type="river_crossing", seed=777)

    t0 = time.time()
    nans_found = 0
    min_mass = float("inf")
    max_cell_mass = 0.0
    min_T = float("inf")
    max_T = 0.0

    # Track projection stats across buckets (first 1k, mid, last 1k)
    proj_act_all = 0.0
    proj_act_sig = 0.0  # |proj - raw| > 1e-2 (material intervention vs tiny softplus tail)
    proj_act_flux = 0.0
    proj_act_heat = 0.0
    proj_act_rxn = 0.0
    proj_l1_sum = 0.0
    early_proj_act = 0.0
    late_proj_act = 0.0
    early_proj_sig = 0.0
    late_proj_sig = 0.0

    for t in range(1, total_steps + 1):
        # Periodic player Fire & Frost bolts every 100 steps (simulating continuous gameplay!)
        if t % 100 == 0:
            fe, fm = sample_bolts(1, 128, 128, device, prob=1.0, max_bolts=1)
        else:
            fe, fm = None, None

        raw = model(encode_state(state, fe, fm))
        state, proj, active = project_and_step(state, raw, fe, fm)

        diff = (raw - proj).abs()
        act_all = float(active.float().mean().item())
        act_sig = float((diff > 1e-2).float().mean().item())
        proj_act_all += act_all
        proj_act_sig += act_sig
        proj_act_flux += float(active[:, 0:8].float().mean().item())
        proj_act_heat += float(active[:, 8:10].float().mean().item())
        proj_act_rxn += float(active[:, 10:17].float().mean().item())
        proj_l1_sum += float(diff.mean().item())

        if t <= 1000:
            early_proj_act += act_all
            early_proj_sig += act_sig
        if t > total_steps - 1000:
            late_proj_act += act_all
            late_proj_sig += act_sig

        if t % 500 == 0:
            if (not torch.isfinite(state.mass).all()) or (not torch.isfinite(state.energy).all()):
                nans_found += 1
            min_mass = min(min_mass, float(state.mass.min().item()))
            max_cell_mass = max(max_cell_mass, float(state.mass.sum(dim=1).max().item()))
            C_now, T_now = compute_C_and_T(state.mass, state.energy)
            valid = C_now > 1e-6
            if valid.any():
                min_T = min(min_T, float(T_now[valid].min().item()))
                max_T = max(max_T, float(T_now[valid].max().item()))
            if t % 2500 == 0:
                print(
                    f"  step {t:5d}/{total_steps} | proj_act(>1e-4)={proj_act_all/t*100:.2f}% "
                    f"proj_sig(>1e-2)={proj_act_sig/t*100:.2f}% | "
                    f"max_m={max_cell_mass:.1f} | T=[{min_T:.0f}, {max_T:.0f}] cK | {time.time()-t0:.1f}s"
                )

    elapsed = time.time() - t0
    res = {
        "steps": total_steps,
        "elapsed_s": elapsed,
        "steps_per_sec": total_steps / elapsed,
        "nans_or_infs": nans_found,
        "min_mass": min_mass,
        "max_cell_total_mass": max_cell_mass,
        "min_T_cK": min_T,
        "max_T_cK": max_T,
        "proj_active_pct_overall": (proj_act_all / total_steps) * 100.0,
        "proj_active_pct_sig_1e2": (proj_act_sig / total_steps) * 100.0,
        "proj_active_pct_flux": (proj_act_flux / total_steps) * 100.0,
        "proj_active_pct_heat": (proj_act_heat / total_steps) * 100.0,
        "proj_active_pct_rxn": (proj_act_rxn / total_steps) * 100.0,
        "proj_active_pct_first_1k": (early_proj_act / 1000.0) * 100.0,
        "proj_active_pct_last_1k": (late_proj_act / 1000.0) * 100.0,
        "proj_sig_pct_first_1k": (early_proj_sig / 1000.0) * 100.0,
        "proj_sig_pct_last_1k": (late_proj_sig / 1000.0) * 100.0,
        "proj_mean_l1": proj_l1_sum / total_steps,
        "pass": bool(
            nans_found == 0
            and min_mass >= 0.0
            and max_cell_mass <= H_CAP + 1e-3
            and min_T >= T_MIN - 1.0
            and max_T <= T_MAX + 1.0
        ),
    }
    print(
        f"  10k Rollout DONE in {elapsed:.1f}s ({res['steps_per_sec']:.0f} steps/s) | "
        f"NaNs={nans_found} | proj_act(>1e-4)={res['proj_active_pct_overall']:.2f}% "
        f"[flux={res['proj_active_pct_flux']:.2f}%, heat={res['proj_active_pct_heat']:.2f}%, rxn={res['proj_active_pct_rxn']:.2f}%] | "
        f"proj_sig(>1e-2)={res['proj_active_pct_sig_1e2']:.2f}% "
        f"(first1k={res['proj_sig_pct_first_1k']:.2f}%, last1k={res['proj_sig_pct_last_1k']:.2f}%) | "
        f"PASS={res['pass']}"
    )
    return res


@torch.no_grad()
def eval_g4_latency(model: WorldModel, device: torch.device) -> dict:
    """G4: GPU latency (p50/p99) at 128x128 & 256x256 across batch sizes, plus full step & ORT CPU."""
    print("\n=== G4: Inference & Full-Step Latency Benchmarks ===")
    model.eval()
    bench = {"gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", "cuda_fp32": {}, "cuda_fp16": {}, "full_step_fp32": {}, "ort_cpu": {}}

    def _time_cuda(fn, warmup=20, iters=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        arr = np.array(times)
        return {"p50_ms": float(np.percentile(arr, 50)), "p99_ms": float(np.percentile(arr, 99)), "mean_ms": float(arr.mean())}

    # 1. Pure model inference CUDA FP32 & FP16
    for (H, W) in [(128, 128), (256, 256)]:
        for B in [1, 4, 8, 16]:
            key = f"B{B}_{H}x{W}"
            x = torch.randn(B, 18, H, W, device=device, dtype=torch.float32)
            t32 = _time_cuda(lambda: model(x))
            bench["cuda_fp32"][key] = t32

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                t16 = _time_cuda(lambda: model(x))
            bench["cuda_fp16"][key] = t16
            print(
                f"  [CUDA {key:11s}] FP32 p50={t32['p50_ms']:.3f} ms (p99={t32['p99_ms']:.3f} ms) | "
                f"FP16 p50={t16['p50_ms']:.3f} ms (p99={t16['p99_ms']:.3f} ms)"
            )

    # 2. Full physics step (encode + NN + project_and_step) on GPU
    for (H, W) in [(128, 128), (256, 256)]:
        for B in [1, 4]:
            key = f"B{B}_{H}x{W}"
            st = make_scene(B, H, W, device, scene_type="mixed", seed=901)
            t_full = _time_cuda(lambda: project_and_step(st, model(encode_state(st))), warmup=10, iters=60)
            bench["full_step_fp32"][key] = t_full
            print(f"  [FullStep {key:11s}] p50={t_full['p50_ms']:.3f} ms (p99={t_full['p99_ms']:.3f} ms) vs 100 ms budget")

    # 3. ONNX Runtime CPU latency (B=1)
    onnx_path = OUT_DIR / "world_model.onnx"
    if onnx_path.exists():
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        for (H, W) in [(128, 128), (256, 256)]:
            x_np = np.random.randn(1, 18, H, W).astype(np.float32)
            for _ in range(5):
                sess.run(["deltas"], {"state": x_np})
            times = []
            for _ in range(25):
                t0 = time.perf_counter()
                sess.run(["deltas"], {"state": x_np})
                times.append((time.perf_counter() - t0) * 1000.0)
            arr = np.array(times)
            bench["ort_cpu"][f"B1_{H}x{W}"] = {
                "p50_ms": float(np.percentile(arr, 50)),
                "p99_ms": float(np.percentile(arr, 99)),
                "mean_ms": float(arr.mean()),
            }
            print(f"  [ORT-CPU B1_{H}x{W}] p50={np.percentile(arr, 50):.2f} ms (p99={np.percentile(arr, 99):.2f} ms)")

    return bench


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(OUT_DIR / "world_model.pt", map_location=device, weights_only=True)
    model = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded world_model.pt ({count_parameters(model):,} params) on {device}")

    g1 = eval_g1_invariants(model, device)
    g2 = eval_g2_fidelity(model, device)
    g3 = eval_g3_long_rollout(model, device, total_steps=10000)
    g4 = eval_g4_latency(model, device)

    eco_path = OUT_DIR / "eco_eval_summary.json"
    g5b = json.loads(eco_path.read_text()) if eco_path.exists() else None

    full_results = {
        "G1_invariants": g1,
        "G2_fidelity": g2,
        "G3_long_rollout_10k": g3,
        "G4_latency": g4,
        "G5B_ecosystem": g5b,
    }
    out_json = OUT_DIR / "gate_results.json"
    out_json.write_text(json.dumps(full_results, indent=2))
    print(f"\nSaved all gate results to {out_json}")


if __name__ == "__main__":
    main()
