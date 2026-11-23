"""
Train the Frostmarsh Neural World Model (6x3x3 Residual CNN, width=48, RF=13)
and export a self-contained `world_model.onnx` (opset 17, dynamic B/H/W).

Training features:
- Online GPU replay pool of diverse 64x64 and 128x128 Frostmarsh scenes
- Direct Huber supervision on (flux_e, flux_s, heat_e, heat_s, rxn_extent)
- Multi-step pushforward rollout loss ramping k = 1 -> 2 -> 4 -> 8
- Small Gaussian input noise during rollout for long-horizon stability
- Projection penalty ||Y_raw - Y_proj||_1 so raw outputs stay inside invariants
- ONNX opset 17 export (single self-contained file) + ORT verification at 128x128 & 256x256
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from model import WorldModel, count_parameters
from sim_teacher import (
    RefSim,
    SimState,
    compute_C_and_T,
    encode_state,
    make_scene,
    project_and_step,
    sample_bolts,
)

OUT_DIR = Path("/home/gooning/.t3/scratch/frostmarsh-ml")


def delta_huber_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Channel-balanced Huber loss with active-region weighting so sparse reactions never collapse."""
    loss_flux = F.smooth_l1_loss(pred[:, 0:8], target[:, 0:8], beta=0.02)
    loss_heat = F.smooth_l1_loss(pred[:, 8:10] / 10.0, target[:, 8:10] / 10.0, beta=0.05)
    loss_rxn = F.smooth_l1_loss(pred[:, 10:17], target[:, 10:17], beta=0.02)

    # Active-pixel supervision for flux, heat, and each of the 7 sparse reactions individually
    act_f = target[:, 0:8].abs() > 1e-3
    l_flux_act = F.smooth_l1_loss(pred[:, 0:8][act_f], target[:, 0:8][act_f], beta=0.03) if act_f.any() else pred.new_tensor(0.0)

    act_h = target[:, 8:10].abs() > 1e-2
    l_heat_act = F.smooth_l1_loss(pred[:, 8:10][act_h] / 10.0, target[:, 8:10][act_h] / 10.0, beta=0.05) if act_h.any() else pred.new_tensor(0.0)

    l_rxn_act = pred.new_tensor(0.0)
    n_rxn_seen = 0
    for r in range(7):
        g_r = target[:, 10 + r]
        p_r = pred[:, 10 + r]
        m_r = g_r > 1e-3
        if m_r.any():
            l_rxn_act = l_rxn_act + F.smooth_l1_loss(p_r[m_r], g_r[m_r], beta=0.05)
            n_rxn_seen += 1
    if n_rxn_seen > 0:
        l_rxn_act = l_rxn_act / n_rxn_seen

    total = (
        12.0 * loss_flux + 3.0 * l_flux_act
        + 4.0 * loss_heat + 1.5 * l_heat_act
        + 8.0 * loss_rxn + 4.0 * l_rxn_act
    )
    return total, {
        "flux": float(loss_flux.item()),
        "heat": float(loss_heat.item()),
        "rxn": float(loss_rxn.item()),
        "rxn_act": float(l_rxn_act.item()),
    }


def state_rollout_loss(pred: SimState, gt: SimState) -> torch.Tensor:
    """Multi-step state matching loss (mass, temperature, liquid flow)."""
    l_mass = F.smooth_l1_loss(pred.mass / 1000.0, gt.mass / 1000.0, beta=0.02)
    _, T_pred = compute_C_and_T(pred.mass, pred.energy)
    _, T_gt = compute_C_and_T(gt.mass, gt.energy)
    l_temp = F.smooth_l1_loss((T_pred - 29000.0) / 10000.0, (T_gt - 29000.0) / 10000.0, beta=0.02)
    l_flow = 0.5 * (
        F.smooth_l1_loss(pred.flow_e / 1000.0, gt.flow_e / 1000.0, beta=0.02)
        + F.smooth_l1_loss(pred.flow_s / 1000.0, gt.flow_s / 1000.0, beta=0.02)
    )
    return 6.0 * l_mass + 4.0 * l_temp + 2.0 * l_flow


def proj_penalty(raw: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Weighted L1 projection penalty ||Y_raw - Y_proj||_1 (gentle on rxn to prevent softplus collapse)."""
    diff = (raw - proj).abs()
    p_flux = diff[:, 0:8].mean()
    p_heat = (diff[:, 8:10] / 10.0).mean()
    p_rxn = diff[:, 10:17].mean()
    return 4.0 * p_flux + 1.0 * p_heat + 0.5 * p_rxn


class ReplayPool:
    """GPU pool of diverse Frostmarsh scenes stepped continuously with RefSim."""

    def __init__(self, size: int, H: int, W: int, device: torch.device, seed: int = 42):
        self.size = size
        self.H = H
        self.W = W
        self.device = device
        self.state = make_scene(size, H, W, device, scene_type="mixed", seed=seed)
        self.age = torch.zeros(size, dtype=torch.long, device=device)

        # Warm up pool to varied ages (0..80 ticks) so training starts on rich dynamics
        with torch.no_grad():
            for t in range(40):
                fe, fm = sample_bolts(size, H, W, device, prob=0.25, max_bolts=2)
                nxt, _ = RefSim.step(self.state, fe, fm)
                # Keep some scenes young by resetting subsets at different steps
                keep = torch.rand(size, device=device) > 0.04
                if not keep.all():
                    fresh = make_scene(size, H, W, device, scene_type="mixed")
                    m = keep.view(size, 1, 1, 1)
                    self.state = SimState(
                        mass=torch.where(m, nxt.mass, fresh.mass),
                        energy=torch.where(m, nxt.energy, fresh.energy),
                        flow_e=torch.where(m, nxt.flow_e, fresh.flow_e),
                        flow_s=torch.where(m, nxt.flow_s, fresh.flow_s),
                        elevation=torch.where(m, nxt.elevation, fresh.elevation),
                    )
                    self.age = torch.where(keep, self.age + 1, torch.zeros_like(self.age))
                else:
                    self.state = nxt
                    self.age += 1

    @torch.no_grad()
    def sample_batch(self, batch_size: int) -> SimState:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return SimState(
            mass=self.state.mass[idx].clone(),
            energy=self.state.energy[idx].clone(),
            flow_e=self.state.flow_e[idx].clone(),
            flow_s=self.state.flow_s[idx].clone(),
            elevation=self.state.elevation[idx].clone(),
        )

    @torch.no_grad()
    def advance(self, steps: int = 2, reset_prob: float = 0.06):
        for _ in range(steps):
            fe, fm = sample_bolts(self.size, self.H, self.W, self.device, prob=0.28, max_bolts=2)
            self.state, _ = RefSim.step(self.state, fe, fm)
            self.age += 1

        # Reset old or random scenes to keep initial transients fresh
        reset_mask = (self.age > 240) | (torch.rand(self.size, device=self.device) < reset_prob)
        if reset_mask.any():
            n_reset = int(reset_mask.sum().item())
            fresh = make_scene(n_reset, self.H, self.W, self.device, scene_type="mixed")
            self.state.mass[reset_mask] = fresh.mass
            self.state.energy[reset_mask] = fresh.energy
            self.state.flow_e[reset_mask] = fresh.flow_e
            self.state.flow_s[reset_mask] = fresh.flow_s
            self.state.elevation[reset_mask] = fresh.elevation
            self.age[reset_mask] = 0


def export_and_verify_onnx(model: WorldModel, onnx_path: Path, device: torch.device) -> dict:
    """Export single self-contained ONNX opset 17 file and verify with ORT at 128x128 & 256x256."""
    model_cpu = WorldModel().eval().cpu()
    model_cpu.load_state_dict(model.state_dict())

    dummy = torch.randn(1, 18, 64, 64, dtype=torch.float32)

    # Remove any stale external data file
    data_file = Path(str(onnx_path) + ".data")
    if data_file.exists():
        data_file.unlink()
    if onnx_path.exists():
        onnx_path.unlink()

    # Export with opset 17 (use legacy exporter if dynamo creates external data, or repack cleanly)
    torch.onnx.export(
        model_cpu,
        (dummy,),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["deltas"],
        dynamic_axes={
            "state": {0: "batch", 2: "height", 3: "width"},
            "deltas": {0: "batch", 2: "height", 3: "width"},
        },
        dynamo=False,
    )

    # Ensure strictly single self-contained file (no .data split)
    proto = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(proto, str(onnx_path), save_as_external_data=False)
    if data_file.exists():
        data_file.unlink()

    onnx.checker.check_model(str(onnx_path))
    assert not data_file.exists(), "External .data file must not exist!"

    # Verify with onnxruntime on 128x128 and 256x256
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    assert in_meta.name == "state", f"Unexpected input name: {in_meta.name}"
    assert out_meta.name == "deltas", f"Unexpected output name: {out_meta.name}"

    verify_stats = {
        "path": str(onnx_path),
        "size_bytes": onnx_path.stat().st_size,
        "opset": 17,
        "input_name": in_meta.name,
        "input_shape": in_meta.shape,
        "output_name": out_meta.name,
        "output_shape": out_meta.shape,
    }

    for (B, H, W) in [(1, 128, 128), (2, 256, 256)]:
        st = make_scene(B, H, W, torch.device("cpu"), scene_type="mixed", seed=123 + H)
        fe, fm = sample_bolts(B, H, W, torch.device("cpu"), prob=1.0, max_bolts=2)
        x = encode_state(st, fe, fm)
        with torch.no_grad():
            y_pt = model_cpu(x).numpy()
        y_ort = sess.run(["deltas"], {"state": x.numpy()})[0]
        max_diff = float(np.max(np.abs(y_pt - y_ort)))
        assert y_ort.shape == (B, 17, H, W), f"Shape mismatch: {y_ort.shape}"
        assert np.all(np.isfinite(y_ort)), "Non-finite ORT output"
        assert max_diff < 1e-4, f"PyTorch vs ORT mismatch {max_diff} at {H}x{W}"
        verify_stats[f"ort_max_diff_{H}x{W}"] = max_diff
        print(f"  [ONNX OK] {B}x18x{H}x{W} -> {y_ort.shape}, max_abs_diff={max_diff:.2e}")

    return verify_stats


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Frostmarsh WorldModel on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device)
    n_params = count_parameters(model)
    print(f"WorldModel parameters: {n_params:,} (6x 3x3 Conv2d, width=48, RF=13)")

    # Dual pools: 64x64 (fast high-batch diversity) + 128x128 (larger spatial structures)
    pool64 = ReplayPool(size=128, H=64, W=64, device=device, seed=100)
    pool128 = ReplayPool(size=32, H=128, W=128, device=device, seed=200)

    # Curriculum schedule: ramp rollout horizon k = 1 -> 2 -> 4 -> 8
    stages = [
        # (num_steps, rollout_k, noise_std, lr, batch64, batch128)
        (700, 1, 0.002, 2.5e-3, 24, 6),
        (600, 2, 0.004, 1.5e-3, 16, 4),
        (500, 4, 0.005, 8.0e-4, 12, 4),
        (400, 8, 0.006, 3.0e-4, 8, 2),
    ]
    total_steps = sum(s[0] for s in stages)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=8e-5)

    t0 = time.time()
    global_step = 0
    history = []

    for stage_idx, (n_steps, k_roll, noise_std, stage_lr, b64, b128) in enumerate(stages):
        # Smoothly cap lr if needed while letting cosine schedule progress
        print(f"\n--- Stage {stage_idx + 1}: steps={n_steps}, k={k_roll}, noise={noise_std}, b64={b64}, b128={b128} ---")
        for s in range(n_steps):
            global_step += 1

            # Every 4th step sample 128x128 crop, otherwise 64x64 crop
            use_128 = (global_step % 4 == 0)
            if use_128:
                init_state = pool128.sample_batch(b128)
                B, H, W = b128, 128, 128
            else:
                init_state = pool64.sample_batch(b64)
                B, H, W = b64, 64, 64

            # Pre-generate forcing sequence for k_roll steps (higher bolt prob so Freeze/Boil/Burn stay well-represented)
            fe_seq, fm_seq = [], []
            for _ in range(k_roll):
                fe, fm = sample_bolts(B, H, W, device, prob=0.55, max_bolts=3)
                fe_seq.append(fe)
                fm_seq.append(fm)

            gt_state = init_state
            pred_state = init_state

            total_loss = torch.tensor(0.0, device=device)
            acc_delta = 0.0
            acc_state = 0.0
            acc_proj = 0.0
            acc_active = 0.0
            comp_last = {"flux": 0.0, "heat": 0.0, "rxn": 0.0}

            for step_i in range(k_roll):
                fe = fe_seq[step_i]
                fm = fm_seq[step_i]

                with torch.no_grad():
                    gt_next, _ = RefSim.step(gt_state, fe, fm)
                    # DAgger teacher delta evaluated on student's current state
                    _, teacher_deltas = RefSim.step(pred_state.detach(), fe, fm)

                x = encode_state(pred_state, fe, fm)
                if noise_std > 0.0:
                    # Add small Gaussian noise to continuous state channels (0..14), keep rock_wall/forcing exact
                    noise = torch.zeros_like(x)
                    noise[:, :14] = torch.randn_like(x[:, :14]) * noise_std
                    x_in = x + noise
                else:
                    x_in = x

                raw_deltas = model(x_in)
                pred_next, proj_deltas, active_mask = project_and_step(pred_state, raw_deltas, fe, fm)

                l_delta, comp = delta_huber_loss(raw_deltas, teacher_deltas)
                l_state = state_rollout_loss(pred_next, gt_next)
                l_proj = proj_penalty(raw_deltas, proj_deltas)

                # Weight later rollout steps slightly higher for stability
                w_step = 1.0 + 0.15 * step_i
                step_loss = w_step * (l_delta + (0.6 if k_roll > 1 else 0.25) * l_state + 0.5 * l_proj)
                total_loss = total_loss + step_loss / k_roll

                acc_delta += float(l_delta.item()) / k_roll
                acc_state += float(l_state.item()) / k_roll
                acc_proj += float(l_proj.item()) / k_roll
                acc_active += float(active_mask.float().mean().item()) / k_roll
                comp_last = comp

                gt_state = gt_next
                # Truncate gradient across very long chains? k<=8 fits easily in GB10 memory!
                pred_state = pred_next

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            # Periodically advance replay pools
            if global_step % 5 == 0:
                pool64.advance(steps=2, reset_prob=0.05)
            if global_step % 12 == 0:
                pool128.advance(steps=2, reset_prob=0.08)

            if global_step % 100 == 0 or s == n_steps - 1:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                rec = {
                    "step": global_step,
                    "k": k_roll,
                    "loss": float(total_loss.item()),
                    "delta": acc_delta,
                    "state": acc_state,
                    "proj": acc_proj,
                    "proj_active_pct": acc_active * 100.0,
                    "flux": comp_last["flux"],
                    "heat": comp_last["heat"],
                    "rxn": comp_last["rxn"],
                    "lr": lr_now,
                    "elapsed_s": elapsed,
                }
                history.append(rec)
                print(
                    f"step {global_step:4d}/{total_steps} (k={k_roll}) | "
                    f"loss={rec['loss']:.4f} [d={acc_delta:.4f} s={acc_state:.4f} p={acc_proj:.4f}] | "
                    f"flux={comp_last['flux']:.5f} heat={comp_last['heat']:.5f} rxn={comp_last['rxn']:.5f} rxn_act={comp_last.get('rxn_act', 0.0):.4f} | "
                    f"proj_act={acc_active*100:.2f}% | lr={lr_now:.2e} | {elapsed:.1f}s"
                )

    train_time = time.time() - t0
    ckpt_path = OUT_DIR / "world_model.pt"
    torch.save({"state_dict": model.state_dict(), "n_params": n_params, "history": history}, ckpt_path)
    print(f"\nSaved PyTorch checkpoint to {ckpt_path} ({train_time:.1f}s total)")

    onnx_path = OUT_DIR / "world_model.onnx"
    print(f"Exporting ONNX to {onnx_path} ...")
    onnx_stats = export_and_verify_onnx(model, onnx_path, device)

    summary = {
        "n_params": n_params,
        "train_time_s": train_time,
        "total_steps": total_steps,
        "final_metrics": history[-1],
        "onnx": onnx_stats,
    }
    (OUT_DIR / "world_train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"World model training + ONNX export complete! Size: {onnx_stats['size_bytes']/1024:.1f} KiB")


if __name__ == "__main__":
    main()
