#!/usr/bin/env python3
"""
Part B: Sparse Active-Tile WorldModel Benchmark on NVIDIA GB10.

Benchmarks WorldModel FP16 (world_model.pt) across:
  1. Full dense 256x256x4 (262,144 voxels) — eager FP16 and CUDA Graph FP16
  2. Full dense 512x512x4 (1,048,576 voxels) — eager FP16 and CUDA Graph FP16
  3. Sparse active-tile batching (16x16 active tiles + halo=6 matching RF=13) at
     20%, 30%, and 50% active tile ratios on both 256x256x4 and 512x512x4:
       - Standard padded 28x28 tile batching (WorldModel forward unchanged)
       - Shrinking-halo valid-conv tile batching (exact mathematically identical
         6x3x3 unpadded pyramid: 28->26->24->22->20->18->16, zero intermediate pads)
       - CUDA Graph versions of sparse active-tile batching

Measures p50_ms, p95_ms, speedup vs dense, and exact numerical max_abs_diff on
interior active cells vs the dense pass (in both FP16 and FP32).

Saves results to:
  /home/gooning/.t3/scratch/frostmarsh-ml/SPARSE_WORLD_BENCH.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import WorldModel

ROOT = Path("/home/gooning/.t3/scratch/frostmarsh-ml")
WORLD_PT = ROOT / "world_model.pt"
OUT_JSON = ROOT / "SPARSE_WORLD_BENCH.json"

TILE_SIZE = 16
HALO = 6  # WorldModel has 6 sequential 3x3 convs => RF = 1 + 6*2 = 13 => exact radius = 6
PATCH_SIZE = TILE_SIZE + 2 * HALO  # 28x28
LAYERS = 4
IN_CH = 18
OUT_CH = 17


def load_world_model(device: torch.device) -> WorldModel:
    wm = WorldModel(in_ch=IN_CH, out_ch=OUT_CH, width=48, num_res=5).to(device).eval()
    ckpt = torch.load(WORLD_PT, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    wm.load_state_dict(sd)
    return wm


def make_realistic_world_tensor(h: int, w: int, layers: int, seed: int, device: torch.device) -> torch.Tensor:
    """Create a realistic 3D Frostmarsh world state tensor [L, 18, H, W] on GPU."""
    rng = np.random.default_rng(seed)
    state = np.zeros((layers, IN_CH, h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = xx / w, yy / h

    rcx = 0.50 + 0.05 * np.sin(ny * 6.28 * 1.3)
    in_river = np.abs(nx - rcx) < 0.08
    state[1, 3] = np.where(~in_river, 0.92, 0.0)  # soil
    state[1, 0] = np.where(in_river, 1.10, 0.0)   # water
    state[1, 13] = np.where(in_river, 0.13, 0.0)
    state[1, 14] = np.where(in_river, 0.15, 0.45)

    # Oil pools & wildfires
    for _ in range(8):
        ox, oy = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
        om = ((nx - ox) ** 2 + (ny - oy) ** 2) < 0.05 ** 2
        state[1, 8][om] = 0.85
        state[1, 0][om] = 0.15
    for _ in range(8):
        fx, fy = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
        fm = ((nx - fx) ** 2 + (ny - fy) ** 2) < 0.04 ** 2
        state[1, 10][fm] = rng.uniform(2.5, 4.5)
        state[1, 7][fm] = 0.35
        state[1, 5][fm] = 0.50

    # Ice & highland bridges
    ice_m = in_river & (np.abs(ny - 0.25) < 0.04)
    state[1, 1][ice_m] = 0.95
    state[1, 10][ice_m] = -0.25
    state[2, 3] = np.where((nx > 0.1) & (nx < 0.9) & (ny > 0.1) & (ny < 0.35), 0.95, 0.0)
    state[2, 5] = np.where((np.abs(ny - 0.67) < 0.025) & (nx > 0.3) & (nx < 0.7), 0.60, 0.0)
    state[0, 0] = np.where(in_river, 1.15, 0.0)
    state[0, 4] = np.where(~in_river, 1.20, 0.0)
    state[3, 2] = np.where(state[1, 10] > 2.0, 0.30, 0.0)

    for l in range(layers):
        state[l, 11] = np.log1p(state[l, :10].sum(axis=0))

    return torch.from_numpy(state).to(device=device, dtype=torch.float32)


def make_active_tile_mask(layers: int, h: int, w: int, ratio: float, seed: int) -> torch.Tensor:
    """
    Build a spatially clustered boolean active tile mask [L, H//16, W//16] with exact active ratio.
    Prefers realistic interior clusters (wildfires/combat/rivers) while reaching exact requested tile count.
    """
    th, tw = h // TILE_SIZE, w // TILE_SIZE
    total_tiles = layers * th * tw
    target_k = max(1, int(round(total_tiles * ratio)))

    rng = np.random.default_rng(seed)
    # Score tiles by smooth clustered activity field (simulating fire/combat/river hotspots)
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    scores = np.zeros((layers, th, tw), dtype=np.float32)
    for l in range(layers):
        # 4 random hotspots per layer
        for _ in range(4):
            cx = rng.uniform(1.5, tw - 2.5)
            cy = rng.uniform(1.5, th - 2.5)
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            scores[l] += np.exp(-d2 / (0.045 * tw * th + 1.0))
        scores[l] += rng.uniform(0.0, 0.05, size=(th, tw))
        # Slight penalty on outer 1-tile boundary so clusters mimic realistic interior gameplay activity
        scores[l, 0, :] -= 0.15
        scores[l, -1, :] -= 0.15
        scores[l, :, 0] -= 0.15
        scores[l, :, -1] -= 0.15

    flat_idx = np.argsort(scores.ravel())[::-1][:target_k]
    mask = np.zeros(total_tiles, dtype=bool)
    mask[flat_idx] = True
    return torch.from_numpy(mask.reshape(layers, th, tw))


def shrinking_halo_forward(wm: WorldModel, patches: torch.Tensor) -> torch.Tensor:
    """
    Exact mathematically equivalent WorldModel forward on [K, 18, 28, 28] patches -> [K, 17, 16, 16].
    Eliminates all 6 intermediate replicate pads and shrinks spatial resolution by 2 per layer
    (28 -> 26 -> 24 -> 22 -> 20 -> 18 -> 16), reducing FLOPs by 42.3% with ZERO approximation!
    """
    h = F.conv2d(patches, wm.stem.conv.weight, wm.stem.conv.bias, padding=0)
    for blk in wm.blocks:
        conv_out = F.conv2d(F.gelu(h), blk.conv.conv.weight, blk.conv.conv.bias, padding=0)
        h = h[:, :, 1:-1, 1:-1] + blk.gamma * conv_out
    raw = wm.head(F.gelu(h)) * wm.out_scale
    lin = raw[:, :10]
    rxn = F.softplus(raw[:, 10:])
    return torch.cat([lin, rxn], dim=1)


def extract_tile_patches(state: torch.Tensor, tile_flat_idx: torch.Tensor) -> torch.Tensor:
    """
    Vectorized zero-loop GPU gather of [K, 18, 28, 28] patches for active tiles.
    state: [L, C, H, W]
    tile_flat_idx: [K] int64 indices into flattened [L * (H//16) * (W//16)] tile grid
    """
    padded = F.pad(state, (HALO, HALO, HALO, HALO), mode="replicate")
    # Unfold spatial dims into [L, C, th, tw, 28, 28] with stride=TILE_SIZE=16
    unfolded = padded.unfold(2, PATCH_SIZE, TILE_SIZE).unfold(3, PATCH_SIZE, TILE_SIZE)
    # Permute to [L, th, tw, C, 28, 28] and flatten tile dims -> [L*th*tw, C, 28, 28]
    l, c, th, tw, ph, pw = unfolded.shape
    tiles_all = unfolded.permute(0, 2, 3, 1, 4, 5).reshape(l * th * tw, c, ph, pw)
    return tiles_all.index_select(0, tile_flat_idx).contiguous()


def scatter_tile_outputs(
    out_buf: torch.Tensor,
    tile_outs: torch.Tensor,
    tile_flat_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized zero-loop GPU scatter of [K, 17, 16, 16] interior tiles back into [L, 17, H, W].
    """
    l, c, h, w = out_buf.shape
    th, tw = h // TILE_SIZE, w // TILE_SIZE
    # Reshape out_buf to [L, th, tw, C, 16, 16] via view+permute? Better: scatter on tile-ordered buffer or index_put
    # View [L, C, th, 16, tw, 16] -> permute to [L, th, tw, C, 16, 16]
    # Note: to scatter in-place without extra copy, maintain tile-major or use a direct view scatter:
    out_tiles = out_buf.view(l, c, th, TILE_SIZE, tw, TILE_SIZE).permute(0, 2, 4, 1, 3, 5).reshape(l * th * tw, c, TILE_SIZE, TILE_SIZE)
    out_tiles.index_copy_(0, tile_flat_idx, tile_outs)
    return out_tiles.view(l, th, tw, c, TILE_SIZE, TILE_SIZE).permute(0, 3, 1, 4, 2, 5).reshape(l, c, h, w)


def build_gather_scatter_maps(
    layers: int, h: int, w: int, tile_mask: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return active flat tile indices plus (l, ty, tx) coordinates on GPU."""
    coords = torch.nonzero(tile_mask, as_tuple=False).to(device)  # [K, 3] -> (l, ty, tx)
    th, tw = h // TILE_SIZE, w // TILE_SIZE
    flat_idx = coords[:, 0] * (th * tw) + coords[:, 1] * tw + coords[:, 2]
    return flat_idx.contiguous(), coords[:, 0], coords[:, 1], coords[:, 2]


def time_gpu_fn(fn, warmup: int = 15, iters: int = 80) -> tuple[float, float]:
    """Time a GPU callable accurately using CUDA events; returns (p50_ms, p95_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return round(float(np.median(times_ms)), 3), round(float(np.percentile(times_ms, 95)), 3)


def capture_cuda_graph(fn, warmup: int = 5) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        fn()
    torch.cuda.synchronize()
    return g


def benchmark_grid(wm_fp32: WorldModel, wm_fp16: WorldModel, h: int, w: int, device: torch.device) -> dict:
    voxels = LAYERS * h * w
    state_fp32 = make_realistic_world_tensor(h, w, LAYERS, seed=100 + h, device=device)
    state_fp16 = state_fp32.half().contiguous()

    # 1. Reference Dense FP32 (strict IEEE FP32, no TF32) & Dense FP16
    with torch.inference_mode():
        prev_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        dense_out_fp32 = wm_fp32(state_fp32)
        torch.backends.cudnn.allow_tf32 = prev_tf32
        dense_out_fp16 = wm_fp16(state_fp16).float()

    # Time Dense FP16 (Eager)
    def run_dense_eager():
        with torch.inference_mode():
            return wm_fp16(state_fp16)

    dense_eager_p50, dense_eager_p95 = time_gpu_fn(run_dense_eager)

    # Time Dense FP16 (CUDA Graph)
    static_in = state_fp16.clone()
    static_out = torch.empty((LAYERS, OUT_CH, h, w), dtype=torch.float16, device=device)

    def run_dense_into():
        static_out.copy_(wm_fp16(static_in))

    dense_graph = capture_cuda_graph(run_dense_into)
    dense_graph_p50, dense_graph_p95 = time_gpu_fn(dense_graph.replay)

    print(
        f"\n=== Grid {h}x{w}x{LAYERS} ({voxels:,} voxels) ===\n"
        f"  Dense FP16 Eager:      p50={dense_eager_p50:.3f} ms, p95={dense_eager_p95:.3f} ms\n"
        f"  Dense FP16 CUDA Graph: p50={dense_graph_p50:.3f} ms, p95={dense_graph_p95:.3f} ms"
    )

    ratios_res = {}
    for ratio in (0.20, 0.30, 0.50):
        tile_mask = make_active_tile_mask(LAYERS, h, w, ratio=ratio, seed=200 + h + int(ratio * 100))
        flat_idx, tl, ty, tx = build_gather_scatter_maps(LAYERS, h, w, tile_mask, device)
        n_active = int(flat_idx.numel())
        total_tiles = LAYERS * (h // TILE_SIZE) * (w // TILE_SIZE)

        # Build pixel-level masks for numerical accuracy verification
        # interior_active_mask: cells belonging to an active tile AND at least HALO cells from global image edge
        active_px = tile_mask.repeat_interleave(TILE_SIZE, dim=1).repeat_interleave(TILE_SIZE, dim=2).to(device)  # [L, H, W]
        interior_mask = torch.zeros_like(active_px)
        interior_mask[:, HALO : h - HALO, HALO : w - HALO] = True
        eval_interior_mask = (active_px & interior_mask).unsqueeze(1).expand(-1, OUT_CH, -1, -1)
        eval_all_active_mask = active_px.unsqueeze(1).expand(-1, OUT_CH, -1, -1)

        # --- Verify Numerical Exactness (Shrinking-Halo vs Dense, both FP16 and FP32) ---
        with torch.inference_mode():
            # Strict IEEE FP32 exactness check (TF32 disabled to confirm mathematical identity)
            prev_tf32 = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = False
            p32 = extract_tile_patches(state_fp32, flat_idx)
            y32_shrink = shrinking_halo_forward(wm_fp32, p32)
            torch.backends.cudnn.allow_tf32 = prev_tf32
            out32_shrink = torch.zeros_like(dense_out_fp32)
            out32_shrink = scatter_tile_outputs(out32_shrink, y32_shrink, flat_idx)
            max_diff_fp32_interior = float((out32_shrink - dense_out_fp32).abs()[eval_interior_mask].max().item())

            # FP16 exactness check (shrinking-halo)
            p16 = extract_tile_patches(state_fp16, flat_idx)
            y16_shrink = shrinking_halo_forward(wm_fp16, p16)
            out16_shrink = torch.zeros((LAYERS, OUT_CH, h, w), dtype=torch.float16, device=device)
            out16_shrink = scatter_tile_outputs(out16_shrink, y16_shrink, flat_idx).float()
            max_diff_fp16_interior = float((out16_shrink - dense_out_fp16).abs()[eval_interior_mask].max().item())
            mean_diff_fp16_interior = float((out16_shrink - dense_out_fp16).abs()[eval_interior_mask].mean().item())
            max_diff_fp16_all_active = float((out16_shrink - dense_out_fp16).abs()[eval_all_active_mask].max().item())

            # Standard padded 28x28 forward check
            y16_padded = wm_fp16(p16)[:, :, HALO : HALO + TILE_SIZE, HALO : HALO + TILE_SIZE].contiguous()
            out16_padded = torch.zeros((LAYERS, OUT_CH, h, w), dtype=torch.float16, device=device)
            out16_padded = scatter_tile_outputs(out16_padded, y16_padded, flat_idx).float()
            max_diff_padded_fp16_interior = float((out16_padded - dense_out_fp16).abs()[eval_interior_mask].max().item())

        # --- Benchmark 1: Shrinking-Halo Sparse Eager FP16 (end-to-end gather + WorldModel + scatter) ---
        out_buf = torch.zeros((LAYERS, OUT_CH, h, w), dtype=torch.float16, device=device)

        def run_sparse_shrink_eager():
            with torch.inference_mode():
                patches = extract_tile_patches(state_fp16, flat_idx)
                y_tiles = shrinking_halo_forward(wm_fp16, patches)
                return scatter_tile_outputs(out_buf, y_tiles, flat_idx)

        sp_eager_p50, sp_eager_p95 = time_gpu_fn(run_sparse_shrink_eager)

        # --- Benchmark 2: Shrinking-Halo Sparse CUDA Graph FP16 (gather + WorldModel + scatter) ---
        sp_graph_out = torch.zeros((LAYERS, OUT_CH, h, w), dtype=torch.float16, device=device)

        def run_sparse_shrink_graph_body():
            patches = extract_tile_patches(static_in, flat_idx)
            y_tiles = shrinking_halo_forward(wm_fp16, patches)
            res = scatter_tile_outputs(sp_graph_out, y_tiles, flat_idx)
            sp_graph_out.copy_(res)

        sp_graph = capture_cuda_graph(run_sparse_shrink_graph_body)
        sp_graph_p50, sp_graph_p95 = time_gpu_fn(sp_graph.replay)

        # --- Benchmark 3: Standard Padded 28x28 Sparse CUDA Graph FP16 (for reference comparison) ---
        pad_graph_out = torch.zeros((LAYERS, OUT_CH, h, w), dtype=torch.float16, device=device)

        def run_sparse_padded_graph_body():
            patches = extract_tile_patches(static_in, flat_idx)
            y_tiles = wm_fp16(patches)[:, :, HALO : HALO + TILE_SIZE, HALO : HALO + TILE_SIZE].contiguous()
            res = scatter_tile_outputs(pad_graph_out, y_tiles, flat_idx)
            pad_graph_out.copy_(res)

        pad_graph = capture_cuda_graph(run_sparse_padded_graph_body)
        pad_graph_p50, pad_graph_p95 = time_gpu_fn(pad_graph.replay)

        speedup_eager_vs_dense_eager = round(dense_eager_p50 / sp_eager_p50, 3)
        speedup_graph_vs_dense_graph = round(dense_graph_p50 / sp_graph_p50, 3)
        speedup_graph_vs_dense_eager = round(dense_eager_p50 / sp_graph_p50, 3)

        key = f"active_{int(ratio * 100)}pct"
        ratios_res[key] = {
            "active_ratio": ratio,
            "active_tiles": n_active,
            "total_tiles": total_tiles,
            "tile_size": TILE_SIZE,
            "halo": HALO,
            "patch_size": PATCH_SIZE,
            "shrinking_halo_eager": {
                "p50_ms": sp_eager_p50,
                "p95_ms": sp_eager_p95,
                "speedup_vs_dense_eager": speedup_eager_vs_dense_eager,
            },
            "shrinking_halo_cuda_graph": {
                "p50_ms": sp_graph_p50,
                "p95_ms": sp_graph_p95,
                "speedup_vs_dense_graph": speedup_graph_vs_dense_graph,
                "speedup_vs_dense_eager": speedup_graph_vs_dense_eager,
            },
            "padded_28x28_cuda_graph": {
                "p50_ms": pad_graph_p50,
                "p95_ms": pad_graph_p95,
                "speedup_vs_dense_graph": round(dense_graph_p50 / pad_graph_p50, 3),
            },
            "numerical_accuracy": {
                "max_abs_diff_fp16_interior": round(max_diff_fp16_interior, 6),
                "mean_abs_diff_fp16_interior": round(mean_diff_fp16_interior, 7),
                "max_abs_diff_fp32_interior": round(max_diff_fp32_interior, 7),
                "max_abs_diff_padded_fp16_interior": round(max_diff_padded_fp16_interior, 6),
                "max_abs_diff_fp16_all_active": round(max_diff_fp16_all_active, 6),
            },
        }
        print(
            f"  [{key}] ({n_active:4d}/{total_tiles:4d} tiles) "
            f"Shrink-Eager={sp_eager_p50:.3f}ms ({speedup_eager_vs_dense_eager:.2f}x) | "
            f"Shrink-Graph={sp_graph_p50:.3f}ms ({speedup_graph_vs_dense_graph:.2f}x vs dense graph, {speedup_graph_vs_dense_eager:.2f}x vs dense eager) | "
            f"Pad28-Graph={pad_graph_p50:.3f}ms | "
            f"max_abs_diff(fp16)={max_diff_fp16_interior:.5f} (fp32={max_diff_fp32_interior:.2e})"
        )

    return {
        "grid": f"{h}x{w}x{LAYERS}",
        "voxels": voxels,
        "dense_fp16_eager": {
            "p50_ms": dense_eager_p50,
            "p95_ms": dense_eager_p95,
        },
        "dense_fp16_cuda_graph": {
            "p50_ms": dense_graph_p50,
            "p95_ms": dense_graph_p95,
        },
        "sparse_active_tile": ratios_res,
    }


def main() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    wm_fp32 = load_world_model(device)
    wm_fp16 = load_world_model(device).half()

    results = {
        "device": torch.cuda.get_device_name(0),
        "model": "WorldModel (6x 3x3 ResCNN, width=48, RF=13)",
        "precision": "fp16",
        "tile_size": TILE_SIZE,
        "halo": HALO,
        "receptive_field": 1 + 6 * 2,
        "benchmarks": {
            "256x256x4": benchmark_grid(wm_fp32, wm_fp16, 256, 256, device),
            "512x512x4": benchmark_grid(wm_fp32, wm_fp16, 512, 512, device),
        },
    }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_JSON} OK!")


if __name__ == "__main__":
    main()
