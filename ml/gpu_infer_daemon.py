#!/usr/bin/env python3
"""
Frostmarsh Ultra-Fast PyTorch CUDA FP16 3D World + Ecosystem + Shared-Latent Neural Bot Daemon.

Serves low-latency GPU inference on TCP_NODELAY and optional Unix domain socket:
  - op=1: WorldModel      ([B, 18, H, W] f32 -> [B, 17, H, W] f32) + ZERO-COPY caches Z_world [B, 48, H, W] in VRAM
          (supports optional SparseWorldRunner active-tile acceleration via --sparse-world!)
  - op=2: EcosystemModel  ([1,  8, H, W] f32 -> [1,  4, H, W] f32) + ZERO-COPY caches Z_eco   [1, 32, H, W] in VRAM!
  - op=4: BotPolicyNet    ([N, 168]      f32 -> [N, 13]       f32) reads cached Z_world & Z_eco ZERO-COPY from VRAM!
  - op=5: BotPolicyInspect([N, 168]      f32 -> [N, 29]       f32) returns 13 action floats + 16 (4x4) Z_world saliency floats!
  - op=6: AcousticStep    (UTF-8 JSON -> UTF-8 JSON) 3D physical acoustics, emergent bot speech bubbles & human chat overrides!
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bot_policy import ACTION_OUT_DIM, BOT_FEAT_DIM, INSPECT_OUT_DIM, BotPolicyNet  # noqa: E402
try:
    from bot_policy_v2 import BotPolicyNetV2  # type: ignore  # noqa: E402
except Exception:
    BotPolicyNetV2 = None  # type: ignore[assignment]
from model import EcosystemModel, WorldModel  # noqa: E402

try:
    from sparse_world_runner import SparseWorldRunner  # type: ignore  # noqa: E402
except Exception:
    SparseWorldRunner = None  # type: ignore[assignment]

try:
    from live_acoustics import LiveAcousticEngine  # type: ignore  # noqa: E402
except Exception:
    LiveAcousticEngine = None  # type: ignore[assignment]

HEADER_STRUCT = struct.Struct("<BBHHHII")
HEADER_SIZE = HEADER_STRUCT.size  # 16 bytes

OP_WORLD_STEP = 1
OP_ECO_STEP = 2
OP_BOT_POLICY = 4
OP_BOT_POLICY_INSPECT = 5
OP_ACOUSTIC_STEP = 6

STATUS_OK = 0
STATUS_BAD_REQUEST = 1
STATUS_INTERNAL_ERROR = 2

EXPECTED_SPECS = {
    OP_WORLD_STEP: (18, 17),
    OP_ECO_STEP: (8, 4),
}


def _load_state_dict(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _recv_exact(sock: socket.socket, buf: memoryview) -> bool:
    n = len(buf)
    got = 0
    while got < n:
        r = sock.recv_into(buf[got:])
        if r == 0:
            if got == 0:
                return False
            raise ConnectionError(f"Unexpected EOF after {got}/{n} bytes")
        got += r
    return True


def _world_forward_with_latent(model: WorldModel, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = model.stem(state)
    for blk in model.blocks:
        h = blk(h)
    z = F.gelu(h)
    raw = model.head(z) * model.out_scale
    out = torch.cat([raw[:, :10], F.softplus(raw[:, 10:])], dim=1)
    return out, z


def _eco_forward_with_latent(model: EcosystemModel, eco_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rho = torch.clamp(eco_state[:, 0:1], min=0.0)
    plant = torch.clamp(eco_state[:, 4:5], min=0.0)
    water_bar = eco_state[:, 5:6]
    rock_wall = eco_state[:, 7:8]
    open_cell = torch.clamp(1.0 - water_bar - rock_wall, 0.0, 1.0)

    h = model.stem(eco_state)
    for blk in model.blocks:
        h = blk(h)
    z = F.gelu(h)
    raw = model.head(z)

    open_e = open_cell * F.pad(open_cell[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    open_s = open_cell * F.pad(open_cell[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    rho_e_nb = F.pad(rho[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    rho_s_nb = F.pad(rho[:, :, 1:, :], (0, 0, 0, 1), value=0.0)

    rate_e = torch.tanh(raw[:, 0:1]) * 0.65
    rate_s = torch.tanh(raw[:, 1:2]) * 0.65
    fe = torch.where(rate_e >= 0.0, rate_e * rho, rate_e * rho_e_nb) * open_e
    fs = torch.where(rate_s >= 0.0, rate_s * rho, rate_s * rho_s_nb) * open_s

    fw_in = F.pad(fe[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
    fn_in = F.pad(fs[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
    out_sum = F.relu(fe) + F.relu(-fw_in) + F.relu(fs) + F.relu(-fn_in)
    dscale = torch.clamp(rho / (out_sum + 1e-6), max=1.0)
    dscale_e = F.pad(dscale[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    dscale_s = F.pad(dscale[:, :, 1:, :], (0, 0, 0, 1), value=0.0)

    fe = torch.where(fe >= 0.0, fe * dscale, fe * dscale_e)
    fs = torch.where(fs >= 0.0, fs * dscale, fs * dscale_s)

    fw_in2 = F.pad(fe[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
    fn_in2 = F.pad(fs[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
    next_rho = torch.clamp(rho + fw_in2 - fe + fn_in2 - fs, min=0.0)
    next_fear = torch.sigmoid(raw[:, 2:3])
    forage = next_rho * plant * torch.sigmoid(raw[:, 3:4]) * 0.6
    next_momentum = next_rho * (2.5 * torch.sigmoid(raw[:, 4:5]))
    return torch.cat([next_rho, next_fear, forage, next_momentum], dim=1), z


class _CUDAGraphRunner:
    """Captures and replays a FP16 CUDA Graph returning BOTH output tensor AND shared VRAM latent trunk."""

    def __init__(
        self,
        op: int,
        model: torch.nn.Module,
        in_ch: int,
        out_ch: int,
        h: int,
        w: int,
        device: torch.device,
        use_fp16: bool,
        batch: int = 1,
    ):
        self.op = op
        self.device = device
        self.use_fp16 = use_fp16
        self.in_shape = (batch, in_ch, h, w)
        self.out_shape = (batch, out_ch, h, w)
        self.static_in = torch.zeros(self.in_shape, dtype=torch.float32, device=device)
        self.host_in = torch.empty(self.in_shape, dtype=torch.float32, pin_memory=True)
        self.host_out = torch.empty(self.out_shape, dtype=torch.float32, pin_memory=True)
        self.host_in_np = self.host_in.numpy()
        self.host_out_np = self.host_out.numpy()
        self.host_in_u8 = self.host_in_np.view(np.uint8).ravel()
        self.host_out_bytes = memoryview(self.host_out_np).cast("B")

        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_out: torch.Tensor | None = None
        self.static_latent: torch.Tensor | None = None
        self._model = model

        if device.type == "cuda":
            self._capture(model)

    def _run_model(self, model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fn = _world_forward_with_latent if self.op == OP_WORLD_STEP else _eco_forward_with_latent
        if self.use_fp16 and self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                y, z = fn(model, x)
            return y.float(), z.half()
        y, z = fn(model, x)
        return y.float(), z.float()

    def _capture(self, model: torch.nn.Module) -> None:
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                _ = self._run_model(model, self.static_in)
        torch.cuda.current_stream(self.device).wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            self.static_out, self.static_latent = self._run_model(model, self.static_in)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize(self.device)
        self.graph = g

    def infer_from_host_in(self) -> tuple[memoryview, torch.Tensor, torch.Tensor, int]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            if self.graph is not None and self.static_out is not None and self.static_latent is not None:
                self.static_in.copy_(self.host_in, non_blocking=True)
                self.graph.replay()
                self.host_out.copy_(self.static_out, non_blocking=False)
                z = self.static_latent
                x_dev = self.static_in
            else:
                x_dev = self.host_in.to(self.device)
                y, z = self._run_model(self._model, x_dev)
                self.host_out.copy_(y.cpu())
        infer_us = int((time.perf_counter() - t0) * 1_000_000)
        return self.host_out_bytes, z, x_dev, infer_us


def _call_with_accepted_kwargs(fn, *pos_args, **cand_kwargs):
    """Invoke `fn` forwarding only keyword arguments present in its signature (or all if **kwargs)."""
    sig = inspect.signature(fn)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return fn(*pos_args, **cand_kwargs)
    filtered = {k: v for k, v in cand_kwargs.items() if k in params}
    return fn(*pos_args, **filtered)


class _SparseWorldAdapter:
    """Wraps Track A's SparseWorldRunner with pinned host I/O buffers matching _CUDAGraphRunner."""

    def __init__(
        self,
        model: torch.nn.Module,
        h: int,
        w: int,
        batch: int,
        device: torch.device,
        use_fp16: bool,
    ):
        assert SparseWorldRunner is not None
        self.device = device
        self.use_fp16 = use_fp16
        self.in_shape = (batch, 18, h, w)
        self.out_shape = (batch, 17, h, w)
        self.static_in = torch.zeros(self.in_shape, dtype=torch.float32, device=device)
        self.host_in = torch.empty(self.in_shape, dtype=torch.float32, pin_memory=(device.type == "cuda"))
        self.host_out = torch.empty(self.out_shape, dtype=torch.float32, pin_memory=(device.type == "cuda"))
        self.host_in_np = self.host_in.numpy()
        self.host_out_np = self.host_out.numpy()
        self.host_in_u8 = self.host_in_np.view(np.uint8).ravel()
        self.host_out_bytes = memoryview(self.host_out_np).cast("B")
        self.runner = _call_with_accepted_kwargs(
            SparseWorldRunner,
            model,
            device=device,
            use_fp16=use_fp16,
            fp16=use_fp16,
            h=h,
            w=w,
            height=h,
            width=w,
            layers=batch,
            batch=batch,
        )

    def infer_from_host_in(self) -> tuple[memoryview, torch.Tensor, torch.Tensor, float, int]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            self.static_in.copy_(self.host_in, non_blocking=(self.device.type == "cuda"))
            res = self.runner.step(self.static_in)
            out_t, z_world, active_frac = res
            self.host_out.copy_(out_t.float(), non_blocking=False)
            if self.use_fp16 and self.device.type == "cuda" and z_world.dtype != torch.float16:
                z_world = z_world.half()
        infer_us = int((time.perf_counter() - t0) * 1_000_000)
        return self.host_out_bytes, z_world, self.static_in, float(active_frac), infer_us


class InferenceEngine:
    def __init__(
        self,
        world_pt: Path,
        eco_pt: Path,
        bot_pt: Path,
        use_fp16: bool = True,
        warmup_shapes: tuple[tuple[int, int], ...] = ((128, 128),),
        use_sparse_world: bool = True,
        acoustic_pt: Path | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")
        self.lock = threading.Lock()

        if self.device.type == "cuda":
            # Disable cudnn.benchmark when sparse dynamic tile batches are enabled to avoid 1s autotune stalls
            torch.backends.cudnn.benchmark = not bool(use_sparse_world and SparseWorldRunner is not None)
            torch.set_float32_matmul_precision("high")

        self.world_model = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).eval()
        self.world_model.load_state_dict(_load_state_dict(world_pt))
        self.world_model.to(self.device)

        self.eco_model = EcosystemModel(in_ch=8, out_ch=4, width=32, num_res=5).eval()
        self.eco_model.load_state_dict(_load_state_dict(eco_pt))
        self.eco_model.to(self.device)

        self.bot_policy: Any = None
        if bot_pt.exists():
            sd = _load_state_dict(bot_pt)
            if BotPolicyNetV2 is not None and any(k.startswith("fine_adapter.") for k in sd):
                bp = BotPolicyNetV2().eval()
                bp.load_state_dict(sd)
                bp.to(self.device)
                self.bot_policy = bp
                print(f"[gpu_infer_daemon] Loaded Dual-Scale BotPolicyNetV2 from {bot_pt}", flush=True)
            else:
                bp = BotPolicyNet().eval()
                bp.load_state_dict(sd)
                bp.to(self.device)
                self.bot_policy = bp
                print(f"[gpu_infer_daemon] Loaded Shared-Latent BotPolicyNet from {bot_pt}", flush=True)

        self.use_sparse_world: bool = bool(use_sparse_world and SparseWorldRunner is not None)
        self._sparse_runners: dict[tuple[int, int, int], _SparseWorldAdapter] = {}

        self.acoustic_engine: Any = None
        ac_path = acoustic_pt if acoustic_pt is not None else (_SCRIPT_DIR / "acoustic_comm_live.pt")
        if LiveAcousticEngine is not None and ac_path.exists():
            try:
                self.acoustic_engine = _call_with_accepted_kwargs(
                    LiveAcousticEngine,
                    ckpt_path=ac_path,
                    checkpoint=ac_path,
                    path=ac_path,
                    device=self.device,
                    use_fp16=self.use_fp16,
                )
                print(f"[gpu_infer_daemon] Loaded LiveAcousticEngine from {ac_path}", flush=True)
            except Exception as exc:
                try:
                    self.acoustic_engine = LiveAcousticEngine(ac_path, device=self.device)
                    print(f"[gpu_infer_daemon] Loaded LiveAcousticEngine from {ac_path}", flush=True)
                except Exception:
                    print(
                        f"[gpu_infer_daemon] LiveAcousticEngine init failed ({exc}); using physical fallback",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.acoustic_engine = None

        self.models = {
            OP_WORLD_STEP: self.world_model,
            OP_ECO_STEP: self.eco_model,
        }
        self._runners: dict[tuple[int, int, int, int], _CUDAGraphRunner] = {}

        # Zero-copy shared VRAM latent & state cache (updated automatically on every world & eco step!)
        self.last_z_world: torch.Tensor = torch.zeros((4, 48, 256, 256), dtype=torch.float16, device=self.device)
        self.last_z_eco: torch.Tensor = torch.zeros((1, 32, 256, 256), dtype=torch.float16, device=self.device)
        self.last_world_state: torch.Tensor | None = None
        self.last_sparse_active_frac: float = 1.0
        self._acoustic_step_counter: int = 0
        self._last_auto_callout_ts: float = 0.0

        for (h, w) in warmup_shapes:
            for op in (OP_WORLD_STEP, OP_ECO_STEP):
                self._get_runner(op, h, w, 1)

    def _get_runner(self, op: int, h: int, w: int, batch: int = 1) -> _CUDAGraphRunner:
        key = (op, batch, h, w)
        runner = self._runners.get(key)
        if runner is None:
            in_ch, out_ch = EXPECTED_SPECS[op]
            runner = _CUDAGraphRunner(
                op=op,
                model=self.models[op],
                in_ch=in_ch,
                out_ch=out_ch,
                h=h,
                w=w,
                device=self.device,
                use_fp16=self.use_fp16,
                batch=batch,
            )
            self._runners[key] = runner
        return runner

    def _get_sparse_runner(self, h: int, w: int, batch: int) -> _SparseWorldAdapter | None:
        if not self.use_sparse_world or SparseWorldRunner is None:
            return None
        key = (batch, h, w)
        adapter = self._sparse_runners.get(key)
        if adapter is None:
            try:
                adapter = _SparseWorldAdapter(
                    model=self.world_model,
                    h=h,
                    w=w,
                    batch=batch,
                    device=self.device,
                    use_fp16=self.use_fp16,
                )
                self._sparse_runners[key] = adapter
                print(
                    f"[gpu_infer_daemon] Activated SparseWorldRunner for {batch}x18x{h}x{w}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[gpu_infer_daemon] SparseWorldRunner unavailable for {batch}x{h}x{w} ({exc}); using dense CUDA Graph",
                    file=sys.stderr,
                    flush=True,
                )
                self.use_sparse_world = False
                return None
        return adapter

    def infer_bytes(self, op: int, h: int, w: int, recv_fn, batch: int = 1) -> tuple[int, memoryview, int]:
        _, out_ch = EXPECTED_SPECS[op]
        with self.lock:
            if op == OP_WORLD_STEP and self.use_sparse_world:
                s_adapter = self._get_sparse_runner(h, w, batch)
                if s_adapter is not None:
                    if not recv_fn(memoryview(s_adapter.host_in_u8)):
                        raise ConnectionError("EOF reading request payload")
                    try:
                        out_view, z_latent, x_dev, active_frac, infer_us = s_adapter.infer_from_host_in()
                        self.last_z_world = z_latent
                        self.last_world_state = x_dev
                        self.last_sparse_active_frac = active_frac
                        return out_ch, memoryview(bytes(out_view)), infer_us
                    except Exception as exc:
                        print(
                            f"[gpu_infer_daemon] SparseWorldRunner step failed ({exc}); falling back to dense",
                            file=sys.stderr,
                            flush=True,
                        )
                        self.use_sparse_world = False
                        # Reuse already-received payload in dense runner
                        runner = self._get_runner(op, h, w, batch)
                        runner.host_in.copy_(s_adapter.host_in)
                        out_view, z_latent, x_dev, infer_us = runner.infer_from_host_in()
                        self.last_z_world = z_latent
                        self.last_world_state = x_dev
                        self.last_sparse_active_frac = 1.0
                        return out_ch, memoryview(bytes(out_view)), infer_us

            runner = self._get_runner(op, h, w, batch)
            if not recv_fn(memoryview(runner.host_in_u8)):
                raise ConnectionError("EOF reading request payload")
            out_view, z_latent, x_dev, infer_us = runner.infer_from_host_in()
            if op == OP_WORLD_STEP:
                self.last_z_world = z_latent
                self.last_world_state = x_dev
                self.last_sparse_active_frac = 1.0
            elif op == OP_ECO_STEP:
                self.last_z_eco = z_latent
            out_copy = bytes(out_view)
        return out_ch, memoryview(out_copy), infer_us

    def _compute_physical_snr_db(
        self,
        sx: float,
        sy: float,
        sz: float,
        lx: float,
        ly: float,
        lz: float,
        volume: float,
        ws_cpu: np.ndarray | None,
    ) -> float:
        """Lightweight 3D physical acoustic propagation (inverse-square + basalt wall occlusion + slab loss + fire/water noise)."""
        vol = max(0.05, min(1.0, float(volume)))
        src_spl = 56.0 + 38.0 * vol  # 56 dB whisper .. 94 dB shout
        dist = math.sqrt((sx - lx) ** 2 + (sy - ly) ** 2 + ((sz - lz) * 6.0) ** 2)
        spread_loss = 20.0 * math.log10(max(1.0, dist * 0.45 + 1.0))

        wall_loss = 0.0
        slab_loss = 0.0
        noise_floor = 42.0

        if ws_cpu is not None and ws_cpu.ndim == 4:
            layers, _, gh, gw = ws_cpu.shape
            l_src = max(0, min(layers - 1, int(round(sz))))
            l_dst = max(0, min(layers - 1, int(round(lz))))

            # Raymarch up to 16 samples along speaker->listener for rock/ice wall crossings
            steps = min(16, max(2, int(dist / 3.0)))
            wall_hits = 0
            for k in range(1, steps):
                u = k / steps
                rx = int(max(0, min(gw - 1, round(sx + (lx - sx) * u))))
                ry = int(max(0, min(gh - 1, round(sy + (ly - sy) * u))))
                rl = max(0, min(layers - 1, int(round(sz + (lz - sz) * u))))
                # ch 4 = rock, ch 1 = ice
                if ws_cpu[rl, 4, ry, rx] > 0.45:
                    wall_hits += 1
            wall_loss = min(36.0, wall_hits * 6.0)

            # Vertical slab attenuation between strata (e.g., highland plateau slab)
            if l_src != l_dst:
                mx = int(max(0, min(gw - 1, round(0.5 * (sx + lx)))))
                my = int(max(0, min(gh - 1, round(0.5 * (sy + ly)))))
                hi_l = max(l_src, l_dst)
                # ch 3 = soil, ch 4 = rock on upper stratum acts as acoustic slab
                if ws_cpu[hi_l, 3, my, mx] > 0.35 or ws_cpu[hi_l, 4, my, mx] > 0.35:
                    slab_loss = 14.0
                else:
                    slab_loss = 3.0

            # Local ambient noise floor at listener (wildfire ch 10 temp, rushing water ch 0)
            ix = int(max(0, min(gw - 1, round(lx))))
            iy = int(max(0, min(gh - 1, round(ly))))
            local_temp = float(ws_cpu[l_dst, 10, iy, ix])
            local_water = float(ws_cpu[l_dst, 0, iy, ix])
            if local_temp > 1.2:
                noise_floor += min(22.0, (local_temp - 1.0) * 8.0)
            if local_water > 0.4:
                noise_floor += 9.0
        elif abs(sz - lz) > 0.5:
            slab_loss = 8.0

        rx_spl = src_spl - spread_loss - wall_loss - slab_loss
        return round(rx_spl - noise_floor, 1)

    def _fallback_acoustic_step(self, req: dict[str, Any]) -> dict[str, Any]:
        """Clean lightweight physical 3D acoustic fallback when LiveAcousticEngine / checkpoint is not loaded."""
        bots = req.get("bots") or []
        listener = req.get("listener") or [128.0, 128.0, 1.0]
        lx = float(listener[0]) if len(listener) > 0 else 128.0
        ly = float(listener[1]) if len(listener) > 1 else 128.0
        lz = float(listener[2]) if len(listener) > 2 else 1.0
        chat = req.get("chat")

        ws_cpu: np.ndarray | None = None
        gw = float(req.get("grid_w", 256))
        gh = float(req.get("grid_h", 256))
        if self.last_world_state is not None:
            ws_cpu = self.last_world_state.detach().float().cpu().numpy()
            gh, gw = float(ws_cpu.shape[2]), float(ws_cpu.shape[3])

        bubbles: list[dict[str, Any]] = []
        overrides: list[dict[str, Any]] = []

        if isinstance(chat, dict) and chat.get("text"):
            text_raw = str(chat.get("text", "")).strip()
            txt = text_raw.lower()
            cx = float(chat.get("x", lx))
            cy = float(chat.get("y", ly))
            cz = float(chat.get("layer", lz))
            vol = float(chat.get("volume", 1.0))
            chat_team = int(chat.get("team", 0))

            # Parse intent keywords -> element, ability, target coordinates
            elem: int | None = None
            if any(k in txt for k in ("freeze", "frost", "ice", "chill", "glacial")):
                elem = 1
            elif any(k in txt for k in ("oil", "slick", "tar", "grease")):
                elem = 2
            elif any(k in txt for k in ("fire", "burn", "ignite", "flame", "meteor", "pyro")):
                elem = 0

            ability: int | None = None
            if any(k in txt for k in ("ult", "nova", "meteor", "surge")):
                ability = 5
            elif any(k in txt for k in ("bridge", "barrage", "sprint", "shield")):
                ability = 4
            elif any(k in txt for k in ("raft", "carve")):
                ability = 3
            elif any(k in txt for k in ("dash", "rush", "charge")):
                ability = 0
            elif any(k in txt for k in ("melee", "slash", "strike")):
                ability = 1
            elif elem is not None or any(k in txt for k in ("attack", "shoot", "cast", "blast")):
                ability = 2

            # Target landmark coordinates
            tx, ty = cx, cy
            loc_lbl = "TARGET"
            if "north" in txt and "bridge" in txt:
                tx, ty, loc_lbl = gw * 0.50, gh * 0.30, "NORTH BRIDGE"
            elif "south" in txt and "bridge" in txt:
                tx, ty, loc_lbl = gw * 0.50, gh * 0.70, "SOUTH BRIDGE"
            elif "north" in txt or "glacier" in txt:
                tx, ty, loc_lbl = gw * 0.50, gh * 0.18, "NORTH"
            elif "south" in txt or "caldera" in txt:
                tx, ty, loc_lbl = gw * 0.50, gh * 0.82, "SOUTH"
            elif "west" in txt or "marsh" in txt:
                tx, ty, loc_lbl = gw * 0.25, gh * 0.50, "WEST"
            elif "east" in txt or "flank" in txt:
                tx, ty, loc_lbl = gw * 0.75, gh * 0.50, "EAST"
            elif any(k in txt for k in ("center", "spire", "relic", "mid", "bridge")):
                tx, ty, loc_lbl = gw * 0.50, gh * 0.50, "SPIRE"
            elif any(k in txt for k in ("help", "group", "rally", "me", "here", "follow")):
                tx, ty, loc_lbl = cx, cy, "YOUR POS"

            verb = {0: "BURNING", 1: "FREEZING", 2: "OILING"}.get(elem, "PUSHING")
            heard_allies: list[tuple[float, dict[str, Any]]] = []
            for b in bots:
                bx = float(b.get("x", gw * 0.5))
                by = float(b.get("y", gh * 0.5))
                bz = float(b.get("z", 1.0))
                b_team = int(b.get("team", 0))
                snr = self._compute_physical_snr_db(cx, cy, cz, bx, by, bz, vol, ws_cpu)
                if b_team == chat_team and snr >= 0.0:
                    dx = tx - bx
                    dy = ty - by
                    mag = math.hypot(dx, dy)
                    mx, my = (dx / mag, dy / mag) if mag > 1.0 else (0.0, 0.0)
                    overrides.append(
                        {
                            "bot_id": int(b.get("id", 0)),
                            "move_x": round(mx, 3),
                            "move_y": round(my, 3),
                            "elem": elem if elem is not None else -1,
                            "ability": ability if ability is not None else 2,
                            "snr_db": snr,
                            "ttl_ticks": 60,
                        }
                    )
                    heard_allies.append((snr, b))

            heard_allies.sort(key=lambda pair: pair[0], reverse=True)
            for snr_to_bot, b in heard_allies[:2]:
                bx = float(b.get("x", gw * 0.5))
                by = float(b.get("y", gh * 0.5))
                bz = float(b.get("z", 1.0))
                reply_snr = self._compute_physical_snr_db(bx, by, bz, lx, ly, lz, 0.85, ws_cpu)
                reply_txt = (
                    f"COPY! {verb} {loc_lbl}"
                    if snr_to_bot >= 6.0
                    else f"STATIC.. {verb} {loc_lbl}"
                )
                bubbles.append(
                    {
                        "speaker_id": int(b.get("id", 0)),
                        "speaker_name": str(b.get("name", "BOT")),
                        "team": int(b.get("team", 0)),
                        "x": bx,
                        "y": by,
                        "z": bz,
                        "snr_db": reply_snr,
                        "symbol_id": int((elem if elem is not None else 3) + 1),
                        "text": reply_txt,
                    }
                )

        # Periodic emergent tactical callout from an active bot (~every 1.8s)
        now = time.monotonic()
        self._acoustic_step_counter += 1
        if bots and not bubbles and (now - self._last_auto_callout_ts) >= 1.8:
            self._last_auto_callout_ts = now
            b = bots[self._acoustic_step_counter % len(bots)]
            bx = float(b.get("x", gw * 0.5))
            by = float(b.get("y", gh * 0.5))
            bz = float(b.get("z", 1.0))
            snr_l = self._compute_physical_snr_db(bx, by, bz, lx, ly, lz, 0.82, ws_cpu)
            callouts = [
                (1, "HOLDING HIGH GROUND!"),
                (2, "FREEZING FJORD CROSSING!"),
                (3, "OIL SLICK ON FLANK!"),
                (4, "PUSHING CENTRAL SPIRE!"),
                (5, "WATERFALL CHOKE HOT!"),
                (6, "WATCH SKYBRIDGE ARCH!"),
            ]
            sym_id, msg = callouts[self._acoustic_step_counter % len(callouts)]
            if bz >= 1.55:
                sym_id, msg = 1, "HIGH GROUND OVERWATCH!"
            elif snr_l < 0.0:
                msg = "...STATIC ON RIDGE..."
            bubbles.append(
                {
                    "speaker_id": int(b.get("id", 0)),
                    "speaker_name": str(b.get("name", "BOT")),
                    "team": int(b.get("team", 0)),
                    "x": bx,
                    "y": by,
                    "z": bz,
                    "snr_db": snr_l,
                    "symbol_id": sym_id,
                    "text": msg,
                }
            )

        return {
            "bubbles": bubbles,
            "overrides": overrides,
        }

    def step_acoustics(self, raw_json_bytes: bytes) -> tuple[bytes, int]:
        req = json.loads(raw_json_bytes.decode("utf-8"))
        with self.lock:
            t0 = time.perf_counter()
            result: dict[str, Any]
            if (
                self.acoustic_engine is not None
                and self.last_world_state is not None
                and self.last_z_world is not None
            ):
                try:
                    raw_bots = req.get("bots") or []
                    bot_states = []
                    bot_pos_map: dict[int, tuple[float, float, float]] = {}
                    for b in raw_bots:
                        bid = int(b.get("id", 0))
                        bx = float(b.get("x", 128.0))
                        by = float(b.get("y", 128.0))
                        bz = float(b.get("z", 1.0))
                        bot_pos_map[bid] = (bx, by, bz)
                        bot_states.append(
                            {
                                "id": bid,
                                "name": str(b.get("name", f"BOT-{bid}")),
                                "team": int(b.get("team", 0)),
                                "cls": int(b.get("cls", 0)),
                                "x": bx,
                                "y": by,
                                "layer": int(round(bz)),
                                "hp": float(b.get("hp", 1.0)),
                            }
                        )
                    lis = req.get("listener") or [128.0, 128.0, 1.0]
                    listener_pos = (
                        float(lis[0]) if len(lis) > 0 else 128.0,
                        float(lis[1]) if len(lis) > 1 else 128.0,
                        float(lis[2]) if len(lis) > 2 else 1.0,
                    )
                    chat = req.get("chat")
                    raw_res = self.acoustic_engine.step_live(
                        w_state=self.last_world_state,
                        z_world=self.last_z_world,
                        bot_states=bot_states,
                        listener_pos=listener_pos,
                        human_chat=chat if isinstance(chat, dict) and chat.get("text") else None,
                    )
                    now_ts = time.perf_counter()
                    has_chat = isinstance(chat, dict) and bool(chat.get("text"))
                    bubbles: list[dict[str, Any]] = []
                    overrides: list[dict[str, Any]] = []

                    ab_name_to_idx = {
                        "dash": 0,
                        "melee": 1,
                        "cast_bolt": 2,
                        "carve_ice": 3,
                        "ability_e": 4,
                        "ult_r": 5,
                    }
                    for ov in raw_res.get("human_overrides") or []:
                        ab_idx = ab_name_to_idx.get(str(ov.get("primary_ability", "")), 2)
                        overrides.append(
                            {
                                "bot_id": int(ov.get("bot_id", 0)),
                                "move_x": float(ov.get("move_x", 0.0)),
                                "move_y": float(ov.get("move_y", 0.0)),
                                "elem": int(ov.get("elem_override", 0)),
                                "ability": ab_idx,
                                "snr_db": float(ov.get("snr_db", 10.0)),
                                "ttl_ticks": 60,
                            }
                        )

                    if has_chat:
                        for rep in (raw_res.get("human_replies") or [])[:3]:
                            bid = int(rep.get("bot_id", 0))
                            bx, by, bz = bot_pos_map.get(bid, listener_pos)
                            bubbles.append(
                                {
                                    "speaker_id": bid,
                                    "speaker_name": str(rep.get("bot_name", "BOT")),
                                    "team": int(rep.get("team", 0)),
                                    "x": bx,
                                    "y": by,
                                    "z": bz,
                                    "snr_db": float(rep.get("snr_db", 0.0)),
                                    "symbol_id": 15 if rep.get("heard") else 0,
                                    "text": str(rep.get("reply_text", ""))[:72],
                                }
                            )

                    if has_chat or (now_ts - self._last_auto_callout_ts >= 2.2):
                        self._last_auto_callout_ts = now_ts
                        for sb in (raw_res.get("speech_bubbles") or [])[: (1 if has_chat else 2)]:
                            bid = int(sb.get("bot_id", 0))
                            pos = sb.get("pos") or list(bot_pos_map.get(bid, listener_pos))
                            txt = str(sb.get("text", ""))
                            # Strip redundant "[Name @ L1 ...] " prefix for clean floating 3D bubbles
                            if ' "' in txt and txt.endswith('"'):
                                txt = txt.split(' "', 1)[1][:-1]
                            elif "] " in txt:
                                txt = txt.split("] ", 1)[1]
                            bubbles.append(
                                {
                                    "speaker_id": bid,
                                    "speaker_name": str(sb.get("bot_name", "BOT")),
                                    "team": int(sb.get("team", 0)),
                                    "x": float(pos[0]),
                                    "y": float(pos[1]),
                                    "z": float(pos[2]) if len(pos) > 2 else 1.0,
                                    "snr_db": float(sb.get("snr_db", 0.0)),
                                    "symbol_id": int(sb.get("symbol_id", 0)),
                                    "text": txt[:72],
                                }
                            )

                    result = {"bubbles": bubbles, "overrides": overrides}
                except Exception as exc:
                    print(
                        f"[gpu_infer_daemon] LiveAcousticEngine.step_live failed ({exc}); using fallback",
                        file=sys.stderr,
                        flush=True,
                    )
                    result = self._fallback_acoustic_step(req)
            else:
                result = self._fallback_acoustic_step(req)
            infer_us = int((time.perf_counter() - t0) * 1_000_000)
            result.setdefault("bubbles", [])
            result.setdefault("overrides", [])
            result["acoustic_ms"] = round(float(result.get("acoustic_ms", infer_us / 1000.0)), 3)
            result["sparse_pct"] = round(self.last_sparse_active_frac * 100.0, 2)
        out_bytes = json.dumps(result, separators=(",", ":")).encode("utf-8")
        return out_bytes, infer_us

    def infer_bots(self, n_bots: int, raw_bytes: bytes, return_saliency: bool = False) -> tuple[bytes, int]:
        if self.bot_policy is None:
            raise RuntimeError("bot_policy.pt not loaded")
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(n_bots, BOT_FEAT_DIM).copy()
        with self.lock:
            t0 = time.perf_counter()
            with torch.inference_mode():
                bf = torch.from_numpy(arr).to(self.device, non_blocking=False)
                if self.use_fp16:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        res = self.bot_policy(
                            self.last_z_world, self.last_z_eco, bf, return_saliency=return_saliency
                        )
                else:
                    res = self.bot_policy(
                        self.last_z_world, self.last_z_eco, bf, return_saliency=return_saliency
                    )
                if return_saliency:
                    actions, saliency = res
                    out_t = torch.cat([actions.float(), saliency.float()], dim=-1)
                else:
                    out_t = res.float()
                out_np = out_t.cpu().numpy().astype(np.float32, copy=False)
            infer_us = int((time.perf_counter() - t0) * 1_000_000)
        return out_np.tobytes(), infer_us


class GPUInferDaemon:
    def __init__(self, host: str, port: int, engine: InferenceEngine, unix_sock: str | None = None):
        self.host = host
        self.port = port
        self.engine = engine
        self.unix_sock_path: Path | None = Path(unix_sock) if unix_sock else None
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._unix_sock: socket.socket | None = None
        self.actual_port: int = port

    def start_listening(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        srv.bind((self.host, self.port))
        srv.listen(32)
        srv.settimeout(0.5)
        self._sock = srv
        self.actual_port = srv.getsockname()[1]

        if self.unix_sock_path is not None:
            self.unix_sock_path.parent.mkdir(parents=True, exist_ok=True)
            if self.unix_sock_path.exists() or self.unix_sock_path.is_symlink():
                self.unix_sock_path.unlink()
            usock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            usock.bind(str(self.unix_sock_path))
            try:
                os.chmod(self.unix_sock_path, 0o666)
            except OSError:
                pass
            usock.listen(32)
            usock.settimeout(0.5)
            self._unix_sock = usock

    def _accept_loop(self, srv_sock: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = srv_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()

    def serve_forever(self) -> None:
        if self._sock is None:
            self.start_listening()
        assert self._sock is not None
        unix_thread: threading.Thread | None = None
        if self._unix_sock is not None:
            unix_thread = threading.Thread(
                target=self._accept_loop, args=(self._unix_sock,), daemon=True
            )
            unix_thread.start()
        self._accept_loop(self._sock)
        if unix_thread is not None:
            unix_thread.join(timeout=1.0)

    def stop(self) -> None:
        self._stop.set()
        for s in (self._sock, self._unix_sock):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        if self.unix_sock_path is not None:
            try:
                if self.unix_sock_path.exists() or self.unix_sock_path.is_symlink():
                    self.unix_sock_path.unlink()
            except OSError:
                pass

    def _send_error(self, conn: socket.socket, status: int, h: int = 0, w: int = 0) -> None:
        conn.sendall(HEADER_STRUCT.pack(status, 0, 0, h, w, 0, 0))

    def _handle_client(self, conn: socket.socket, _addr) -> None:
        hdr_buf = bytearray(HEADER_SIZE)
        hdr_view = memoryview(hdr_buf)
        try:
            if conn.family in (socket.AF_INET, socket.AF_INET6):
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while not self._stop.is_set():
                if not _recv_exact(conn, hdr_view):
                    break
                op, pad_batch, channels, height, width, payload_bytes, _seq = HEADER_STRUCT.unpack(hdr_buf)

                if op == OP_ACOUSTIC_STEP:
                    if payload_bytes == 0 or payload_bytes > 4 * 1024 * 1024:
                        if 0 < payload_bytes <= 16 * 1024 * 1024:
                            _recv_exact(conn, memoryview(bytearray(payload_bytes)))
                        self._send_error(conn, STATUS_BAD_REQUEST, height, width)
                        continue
                    raw_in = bytearray(payload_bytes)
                    if not _recv_exact(conn, memoryview(raw_in)):
                        break
                    try:
                        out_bytes, infer_us = self.engine.step_acoustics(bytes(raw_in))
                    except Exception as exc:
                        print(f"[gpu_infer_daemon] Acoustic step error: {exc}", file=sys.stderr)
                        self._send_error(conn, STATUS_INTERNAL_ERROR, height, width)
                        continue
                    resp_hdr = HEADER_STRUCT.pack(
                        STATUS_OK, 1, 1, 0, 0, len(out_bytes), infer_us
                    )
                    conn.sendall(resp_hdr)
                    conn.sendall(out_bytes)
                    continue

                if op in (OP_BOT_POLICY, OP_BOT_POLICY_INSPECT):
                    inspect = op == OP_BOT_POLICY_INSPECT
                    out_dim = INSPECT_OUT_DIM if inspect else ACTION_OUT_DIM
                    n_bots = int(height)
                    expected_bytes = n_bots * BOT_FEAT_DIM * 4
                    if n_bots <= 0 or width != BOT_FEAT_DIM or payload_bytes != expected_bytes:
                        if 0 < payload_bytes <= 16 * 1024 * 1024:
                            _recv_exact(conn, memoryview(bytearray(payload_bytes)))
                        self._send_error(conn, STATUS_BAD_REQUEST, height, width)
                        continue
                    raw_in = bytearray(payload_bytes)
                    if not _recv_exact(conn, memoryview(raw_in)):
                        break
                    try:
                        out_bytes, infer_us = self.engine.infer_bots(
                            n_bots, bytes(raw_in), return_saliency=inspect
                        )
                    except Exception as exc:
                        print(f"[gpu_infer_daemon] Bot policy error: {exc}", file=sys.stderr)
                        self._send_error(conn, STATUS_INTERNAL_ERROR, height, width)
                        continue
                    resp_hdr = HEADER_STRUCT.pack(
                        STATUS_OK, 1, 1, n_bots, out_dim, len(out_bytes), infer_us
                    )
                    conn.sendall(resp_hdr)
                    conn.sendall(out_bytes)
                    continue

                batch = max(1, int(pad_batch))
                spec = EXPECTED_SPECS.get(op)
                if spec is None or channels != spec[0] or height == 0 or width == 0:
                    if 0 < payload_bytes <= 64 * 1024 * 1024:
                        _recv_exact(conn, memoryview(bytearray(payload_bytes)))
                    self._send_error(conn, STATUS_BAD_REQUEST, height, width)
                    continue

                expected_bytes = batch * channels * height * width * 4
                if payload_bytes != expected_bytes:
                    if 0 < payload_bytes <= 64 * 1024 * 1024:
                        _recv_exact(conn, memoryview(bytearray(payload_bytes)))
                    self._send_error(conn, STATUS_BAD_REQUEST, height, width)
                    continue

                try:
                    out_ch, out_bytes, infer_us = self.engine.infer_bytes(
                        op, height, width, lambda mv: _recv_exact(conn, mv), batch=batch
                    )
                except Exception as exc:
                    print(f"[gpu_infer_daemon] Inference error for op={op} {height}x{width}: {exc}", file=sys.stderr)
                    self._send_error(conn, STATUS_INTERNAL_ERROR, height, width)
                    continue

                resp_hdr = HEADER_STRUCT.pack(
                    STATUS_OK, batch, out_ch, height, width, len(out_bytes), infer_us
                )
                conn.sendall(resp_hdr)
                conn.sendall(out_bytes)
        except (ConnectionError, BrokenPipeError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


def parse_bind(bind_str: str) -> tuple[str, int]:
    host, port_s = bind_str.rsplit(":", 1)
    return host, int(port_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frostmarsh 3D World + Shared-Latent Neural Bot GPU Daemon")
    parser.add_argument("--bind", default="127.0.0.1:14440")
    parser.add_argument(
        "--unix-sock",
        default=None,
        help="Optional Unix domain socket path for zero-overhead local IPC (e.g. /tmp/frostmarsh-gpu.sock)",
    )
    parser.add_argument("--world-model", type=Path, default=_SCRIPT_DIR / "world_model.pt")
    parser.add_argument("--eco-model", type=Path, default=_SCRIPT_DIR / "ecosystem_model.pt")
    parser.add_argument("--bot-policy", type=Path, default=_SCRIPT_DIR / "bot_policy.pt")
    parser.add_argument("--acoustic-ckpt", type=Path, default=_SCRIPT_DIR / "acoustic_comm_live.pt")
    parser.add_argument(
        "--sparse-world",
        action=argparse.BooleanOptionalAction,
        default=(SparseWorldRunner is not None),
        help="Use SparseWorldRunner active-tile acceleration for op=1 when available (default: True if importable)",
    )
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    host, port = parse_bind(args.bind)
    engine = InferenceEngine(
        args.world_model,
        args.eco_model,
        args.bot_policy,
        use_fp16=not args.fp32,
        use_sparse_world=args.sparse_world,
        acoustic_pt=args.acoustic_ckpt,
    )
    daemon = GPUInferDaemon(host, port, engine, unix_sock=args.unix_sock)
    daemon.start_listening()

    def _sig_handler(_signum, _frame):
        daemon.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    dev_name = torch.cuda.get_device_name(0) if engine.device.type == "cuda" else "CPU"
    unix_info = f" | unix={daemon.unix_sock_path}" if daemon.unix_sock_path is not None else ""
    print(
        f"[gpu_infer_daemon] Ready on {host}:{daemon.actual_port}{unix_info} | device={dev_name} | "
        f"fp16={engine.use_fp16} | neural_bots={engine.bot_policy is not None} | "
        f"sparse_world={engine.use_sparse_world} | acoustics={'neural' if engine.acoustic_engine is not None else 'physical-fallback'}",
        flush=True,
    )
    daemon.serve_forever()


if __name__ == "__main__":
    main()
