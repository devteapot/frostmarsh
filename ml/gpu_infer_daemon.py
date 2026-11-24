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
  - op=7: ActuatorPolicy  ([N, 168]      f32 -> [N, 26]       f32) continuous physical actuators (10 floats) + 16 (4x4) Z_world saliency!
  - op=8: WorldModel3D    ([L, 18, H, W] f32 -> [L, 17, H, W] + [L-1, 5, H, W] f32) true 3D sparse/dense horizontal + vertical interface fluxes + Z_world cache!
  - op=9: WorldModelDT    ([L, 18, H, W] + [H, W] local_dt f32 -> [L, 17, H, W] + [L-1, 5, H, W] f32) spatial time-dilation domes!
"""

from __future__ import annotations

import argparse
from collections import deque
import inspect
import json
import math
import os
import queue
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
    from sparse_eco_runner import SparseEcoRunner  # type: ignore  # noqa: E402
except Exception:
    SparseEcoRunner = None  # type: ignore[assignment]

try:
    from live_acoustics import LiveAcousticEngine  # type: ignore  # noqa: E402
except Exception:
    LiveAcousticEngine = None  # type: ignore[assignment]

try:
    from actuator_policy import ActuatorPolicyNet  # type: ignore  # noqa: E402
except Exception:
    ActuatorPolicyNet = None  # type: ignore[assignment]

try:
    from actuator_policy_v3 import (  # type: ignore  # noqa: E402
        ACT3D_DIM,
        ActuatorPolicyNetV3,
        adapt_bot_feats_168,
    )
except Exception:
    ActuatorPolicyNetV3 = None  # type: ignore[assignment]
    adapt_bot_feats_168 = None  # type: ignore[assignment]
    ACT3D_DIM = 12

HEADER_STRUCT = struct.Struct("<BBHHHII")
HEADER_SIZE = HEADER_STRUCT.size  # 16 bytes

OP_WORLD_STEP = 1
OP_ECO_STEP = 2
OP_BOT_POLICY = 4
OP_BOT_POLICY_INSPECT = 5
OP_ACOUSTIC_STEP = 6
OP_ACTUATOR_POLICY = 7
OP_WORLD_3D_STEP = 8
OP_WORLD_DT_STEP = 9
OP_ACTUATOR_V3 = 10

Z_FLUX_CH = 5
ACTUATOR_OUT_DIM = 10
ACTUATOR_INSPECT_OUT_DIM = ACTUATOR_OUT_DIM + 16  # 26 floats per bot
ACTUATOR_V3_INSPECT_OUT_DIM = ACT3D_DIM + 16      # 28 floats per bot

STATUS_OK = 0
STATUS_BAD_REQUEST = 1
STATUS_INTERNAL_ERROR = 2

EXPECTED_SPECS = {
    OP_WORLD_STEP: (18, 17),
    OP_ECO_STEP: (8, 4),
}


def resolve_default_world_model_v5(script_dir: Path = _SCRIPT_DIR) -> Path:
    """Auto-prefer world_model_3d_v8.pt -> world_model_3d_v7.pt -> world_model_3d_v6.pt -> world_model_3d_v5.pt."""
    for name in (
        "world_model_3d_v8.pt",
        "world_model_3d_v7.pt",
        "world_model_3d_v6.pt",
        "world_model_3d_v5.pt",
    ):
        cand = script_dir / name
        if cand.is_file():
            return cand
    return script_dir / "world_model_3d_v5.pt"


def _load_state_dict(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _remap_z_to_4_strata(z: float, num_layers: int) -> float:
    """Remap continuous vertical altitude z in [0..L-1] into canonical 4-strata band_z in [0..3]."""
    if num_layers <= 4:
        return max(0.0, min(float(max(1, num_layers) - 1), float(z)))
    return max(0.0, min(3.0, (float(z) / float(num_layers - 1)) * 3.0))


def _pool_z_to_4_strata(
    tensor_L: torch.Tensor,
    is_world_state: bool | None = None,
) -> torch.Tensor:
    """
    Pool an isotropic high-depth 3D tensor `[L, C, H, W]` (e.g. `L=16` meters) into the
    4 canonical semantic strata expected by `BotPolicyNetV2`, `ActuatorPolicyNet` v2,
    `LiveAcousticEngine`, and `System2DialogueEngine`:
      - band 0 = z[0..L/4]       (cave / subterranean)
      - band 1 = z[L/4..L/2]     (canyon / river / surface)
      - band 2 = z[L/2..3L/4]    (arched bridge / highland)
      - band 3 = z[3L/4..L]      (mountain peaks / sky)

    Uses `mean` for `Z_world` latent & temperature/flows/elevation, `max` (`amax`) for
    physical species masses / log-density / rock-wall flags, and clamped `sum` for
    forcing energy/mass impulses. No-op when `L <= 4`.
    """
    if tensor_L.ndim != 4 or tensor_L.shape[0] <= 4:
        return tensor_L
    L, C, H, W = tensor_L.shape
    world_mode = (C == 18) if is_world_state is None else bool(is_world_state)

    if L % 4 == 0:
        b = tensor_L.reshape(4, L // 4, C, H, W)
        if not world_mode:
            return b.mean(dim=1).contiguous()
        out = b.mean(dim=1)  # temp (10), flow_e/s (12, 13), elev (14)
        out[:, :10] = b[:, :, :10].amax(dim=1)  # physical species masses [0..9]
        if C > 11:
            out[:, 11] = b[:, :, 11].amax(dim=1)  # log total density
        if C > 15:
            out[:, 15] = b[:, :, 15].amax(dim=1)  # solid rock wall flag
        if C > 16:
            out[:, 16] = b[:, :, 16].sum(dim=1).clamp(-10.0, 10.0)  # stratum forcing energy
        if C > 17:
            out[:, 17] = b[:, :, 17].sum(dim=1).clamp(-4.0, 4.0)  # stratum forcing mass
        return out.contiguous()

    chunks = torch.tensor_split(tensor_L, 4, dim=0)
    if not world_mode:
        return torch.stack([c.mean(dim=0) for c in chunks], dim=0).contiguous()
    bands = []
    for c in chunks:
        band = c.mean(dim=0)
        band[:10] = c[:, :10].amax(dim=0)
        if C > 11:
            band[11] = c[:, 11].amax(dim=0)
        if C > 15:
            band[15] = c[:, 15].amax(dim=0)
        if C > 16:
            band[16] = c[:, 16].sum(dim=0).clamp(-10.0, 10.0)
        if C > 17:
            band[17] = c[:, 17].sum(dim=0).clamp(-4.0, 4.0)
        bands.append(band)
    return torch.stack(bands, dim=0).contiguous()


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


class _SparseEcoAdapter:
    """Wraps SparseEcoRunner with pinned host I/O buffers matching _CUDAGraphRunner."""

    def __init__(
        self,
        model: torch.nn.Module,
        h: int,
        w: int,
        batch: int,
        device: torch.device,
        use_fp16: bool,
    ):
        assert SparseEcoRunner is not None
        self.device = device
        self.use_fp16 = use_fp16
        self.in_shape = (batch, 8, h, w)
        self.out_shape = (batch, 4, h, w)
        self.static_in = torch.zeros(self.in_shape, dtype=torch.float32, device=device)
        self.host_in = torch.empty(self.in_shape, dtype=torch.float32, pin_memory=(device.type == "cuda"))
        self.host_out = torch.empty(self.out_shape, dtype=torch.float32, pin_memory=(device.type == "cuda"))
        self.host_in_np = self.host_in.numpy()
        self.host_out_np = self.host_out.numpy()
        self.host_in_u8 = self.host_in_np.view(np.uint8).ravel()
        self.host_out_bytes = memoryview(self.host_out_np).cast("B")
        self.runner = _call_with_accepted_kwargs(
            SparseEcoRunner,
            model,
            device=device,
            use_fp16=use_fp16,
            fp16=use_fp16,
            h=h,
            w=w,
            height=h,
            width=w,
            batch=batch,
        )

    def infer_from_host_in(self) -> tuple[memoryview, torch.Tensor, torch.Tensor, float, int]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            self.static_in.copy_(self.host_in, non_blocking=(self.device.type == "cuda"))
            out_t, z_eco, active_frac = self.runner.step(self.static_in)
            self.host_out.copy_(out_t.float(), non_blocking=False)
            if self.use_fp16 and self.device.type == "cuda" and z_eco.dtype != torch.float16:
                z_eco = z_eco.half()
        infer_us = int((time.perf_counter() - t0) * 1_000_000)
        return self.host_out_bytes, z_eco, self.static_in, float(active_frac), infer_us


class _SparseWorld3DAdapter:
    """Wraps SparseWorld3DRunner with pinned host I/O buffers for op=8 (`OP_WORLD_3D_STEP`)."""

    def __init__(
        self,
        runner_cls: Any,
        model_3d: torch.nn.Module,
        h: int,
        w: int,
        batch: int,
        device: torch.device,
        use_fp16: bool,
        use_sparse: bool = True,
    ):
        self.device = device
        self.use_fp16 = use_fp16
        self.in_shape = (batch, 18, h, w)
        self.horiz_shape = (batch, 17, h, w)
        self.z_flux_layers = max(0, batch - 1)
        self.z_flux_shape = (self.z_flux_layers, Z_FLUX_CH, h, w)
        self.horiz_elems = batch * 17 * h * w
        self.z_flux_elems = self.z_flux_layers * Z_FLUX_CH * h * w

        self.static_in = torch.zeros(self.in_shape, dtype=torch.float32, device=device)
        self.host_in = torch.empty(self.in_shape, dtype=torch.float32, pin_memory=(device.type == "cuda"))
        self.host_out = torch.empty(
            (self.horiz_elems + self.z_flux_elems,),
            dtype=torch.float32,
            pin_memory=(device.type == "cuda"),
        )
        self.host_out_horiz = self.host_out[: self.horiz_elems].view(*self.horiz_shape)
        self.host_out_zflux = (
            self.host_out[self.horiz_elems :].view(*self.z_flux_shape)
            if self.z_flux_elems > 0
            else None
        )
        self.host_in_np = self.host_in.numpy()
        self.host_out_np = self.host_out.numpy()
        self.host_in_u8 = self.host_in_np.view(np.uint8).ravel()
        self.host_out_bytes = memoryview(self.host_out_np).cast("B")
        self.runner = _call_with_accepted_kwargs(
            runner_cls,
            model_3d,
            device=device,
            use_fp16=use_fp16,
            sparse_threshold=(0.30 if use_sparse else 0.0),
            activity_eps=5e-4,
        )

    def infer_from_host_in(self) -> tuple[memoryview, torch.Tensor, torch.Tensor, float, int]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            self.static_in.copy_(self.host_in, non_blocking=(self.device.type == "cuda"))
            horiz_t, z_flux_t, z_world, active_frac = self.runner.step(self.static_in)
            self.host_out_horiz.copy_(horiz_t.float(), non_blocking=False)
            if self.host_out_zflux is not None:
                self.host_out_zflux.copy_(z_flux_t.float(), non_blocking=False)
            if self.use_fp16 and self.device.type == "cuda" and z_world.dtype != torch.float16:
                z_world = z_world.half()
        infer_us = int((time.perf_counter() - t0) * 1_000_000)
        return self.host_out_bytes, z_world, self.static_in, float(active_frac), infer_us


class _SparseWorldDTAdapter:
    """Wraps SparseWorldDTRunner with pinned host I/O buffers for op=9 (`OP_WORLD_DT_STEP`)."""

    def __init__(
        self,
        runner_cls: Any,
        model_dt: torch.nn.Module,
        h: int,
        w: int,
        batch: int,
        device: torch.device,
        use_fp16: bool,
        use_sparse: bool = True,
    ):
        self.device = device
        self.use_fp16 = use_fp16
        self.x_shape = (batch, 18, h, w)
        self.dt_shape = (h, w)
        self.x_elems = batch * 18 * h * w
        self.dt_elems = h * w
        self.horiz_shape = (batch, 17, h, w)
        self.z_flux_layers = max(0, batch - 1)
        self.z_flux_shape = (self.z_flux_layers, Z_FLUX_CH, h, w)
        self.horiz_elems = batch * 17 * h * w
        self.z_flux_elems = self.z_flux_layers * Z_FLUX_CH * h * w

        self.static_in_x = torch.zeros(self.x_shape, dtype=torch.float32, device=device)
        self.static_in_dt = torch.ones(self.dt_shape, dtype=torch.float32, device=device)
        self.host_in = torch.empty(
            (self.x_elems + self.dt_elems,),
            dtype=torch.float32,
            pin_memory=(device.type == "cuda"),
        )
        self.host_in_x = self.host_in[: self.x_elems].view(*self.x_shape)
        self.host_in_dt = self.host_in[self.x_elems :].view(*self.dt_shape)
        self.host_out = torch.empty(
            (self.horiz_elems + self.z_flux_elems,),
            dtype=torch.float32,
            pin_memory=(device.type == "cuda"),
        )
        self.host_out_horiz = self.host_out[: self.horiz_elems].view(*self.horiz_shape)
        self.host_out_zflux = (
            self.host_out[self.horiz_elems :].view(*self.z_flux_shape)
            if self.z_flux_elems > 0
            else None
        )
        self.host_in_np = self.host_in.numpy()
        self.host_out_np = self.host_out.numpy()
        self.host_in_u8 = self.host_in_np.view(np.uint8).ravel()
        self.host_out_bytes = memoryview(self.host_out_np).cast("B")
        self.runner = _call_with_accepted_kwargs(
            runner_cls,
            model_dt,
            device=device,
            use_fp16=use_fp16,
            sparse_threshold=(0.30 if use_sparse else 0.0),
            activity_eps=5e-4,
        )

    def infer_from_host_in(self) -> tuple[memoryview, torch.Tensor, torch.Tensor, float, int]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            self.static_in_x.copy_(self.host_in_x, non_blocking=(self.device.type == "cuda"))
            self.static_in_dt.copy_(self.host_in_dt, non_blocking=(self.device.type == "cuda"))
            horiz_t, z_flux_t, z_world, active_frac = self.runner.step(
                self.static_in_x, local_dt=self.static_in_dt
            )
            self.host_out_horiz.copy_(horiz_t.float(), non_blocking=False)
            if self.host_out_zflux is not None:
                self.host_out_zflux.copy_(z_flux_t.float(), non_blocking=False)
            if self.use_fp16 and self.device.type == "cuda" and z_world.dtype != torch.float16:
                z_world = z_world.half()
        infer_us = int((time.perf_counter() - t0) * 1_000_000)
        return self.host_out_bytes, z_world, self.static_in_x, float(active_frac), infer_us


class _AsyncSystem2Bridge:
    """
    Strictly non-blocking System-2 `Qwen2.5-1.5B-Instruct` dialogue adapter for `op=6`.
    - Lazily loads `System2DialogueEngine` + `ZWorldToLanguageBridge` in a background thread.
    - `submit_async(...)`: <10 us non-blocking enqueue onto an observation worker queue.
    - `poll_replies()`: <30 us non-blocking poll of completed LLM replies + actuator plans.
    """

    def __init__(self, device: torch.device, use_fp16: bool = True):
        self.device = device
        self.use_fp16 = use_fp16
        self.is_ready: bool = False
        self.load_error: str | None = None
        self._bridge: Any = None
        self._engine: Any = None
        self._bot_profile_cls: Any = None
        self._req_cls: Any = None
        self._obs_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=4)
        self._req_meta: dict[str, dict[str, Any]] = {}
        self._meta_lock = threading.Lock()
        self._stop = threading.Event()
        self._seed_counter = 0

        threading.Thread(target=self._init_and_observe_loop, name="System2SLMInitObs", daemon=True).start()

    def _init_and_observe_loop(self) -> None:
        try:
            from system2_slm import (  # noqa: E402
                BotProfile,
                DialogueRequest,
                System2DialogueEngine,
                ZWorldToLanguageBridge,
            )

            self._bot_profile_cls = BotProfile
            self._req_cls = DialogueRequest
            self._bridge = ZWorldToLanguageBridge(device=self.device, use_fp16=self.use_fp16)
            dev_str = "cuda" if self.device.type == "cuda" else "cpu"
            try:
                self._engine = System2DialogueEngine(
                    device=dev_str, local_files_only=True, start_worker=True
                )
            except Exception:
                self._engine = System2DialogueEngine(
                    device=dev_str, local_files_only=False, start_worker=True
                )
            self.is_ready = True
            print(
                f"[gpu_infer_daemon] System-2 SLM ready ({self._engine.model_id}, "
                f"{self._engine.dtype}, load={self._engine.load_time_s:.2f}s)",
                flush=True,
            )
        except Exception as exc:
            self.load_error = str(exc)
            print(
                f"[gpu_infer_daemon] System-2 SLM init failed ({exc}); running without SLM",
                file=sys.stderr,
                flush=True,
            )
            return

        # Background observation worker: bridges Z_world + w_state + AcousticField3D -> System2DialogueEngine
        while not self._stop.is_set():
            try:
                item = self._obs_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._seed_counter += 1
                b = item["bot"]
                cls_id = int(b.get("class_id", b.get("cls", 0))) % 5
                bot_cls = {
                    0: "Strider",
                    1: "Skywing",
                    2: "Tidekin",
                    3: "Emberstalker",
                    4: "Burrower",
                }.get(cls_id, "Strider")
                hp_frac = float(b.get("hp_frac", b.get("hp", 1.0)))
                hp_val = hp_frac * 100.0 if hp_frac <= 1.5 else hp_frac
                bx = float(b.get("x", 128.0))
                by = float(b.get("y", 128.0))
                bz = int(round(float(b.get("z", 1.0))))
                bot_id_int = int(b.get("id", 0))
                bot_name = str(b.get("name", f"BOT-{bot_id_int}"))
                bot_team = int(b.get("team", 0))
                profile = self._bot_profile_cls(
                    bot_id=str(bot_id_int),
                    name=bot_name,
                    bot_class=bot_cls,
                    team=bot_team,
                    x=bx,
                    y=by,
                    layer=bz,
                    hp=hp_val,
                )
                obs = self._bridge.observe(
                    w_state=item["w_state"],
                    z_world=item["z_world"],
                    bot=profile,
                    speaker_name=item["speaker_name"],
                    speaker_pos=item["speaker_pos"],
                    speaker_volume=item["speaker_volume"],
                    raw_utterance=item["raw_utterance"],
                    seed=self._seed_counter,
                    force_recompute_acoustics=False,
                )
                req = self._req_cls(
                    bot=profile,
                    observation=obs,
                    conversation_id=f"live_team_{bot_team}",
                    max_new_tokens=64,
                )
                with self._meta_lock:
                    self._req_meta[req.request_id] = {
                        "bot_id": bot_id_int,
                        "bot_name": bot_name,
                        "team": bot_team,
                        "x": bx,
                        "y": by,
                        "z": float(b.get("orig_z", b.get("z", 1.0))),
                        "snr_db": float(obs.incoming_snr_db),
                        "grid_w": float(item.get("grid_w", 256.0)),
                        "grid_h": float(item.get("grid_h", 256.0)),
                    }
                    # Prevent unbounded metadata growth
                    if len(self._req_meta) > 32:
                        oldest = next(iter(self._req_meta))
                        self._req_meta.pop(oldest, None)
                self._engine.submit_dialogue(req)
            except Exception as exc:
                print(f"[gpu_infer_daemon] System-2 SLM observe/submit error: {exc}", file=sys.stderr, flush=True)
            finally:
                self._obs_queue.task_done()

    def submit_async(
        self,
        w_state: torch.Tensor,
        z_world: torch.Tensor,
        bot: dict[str, Any],
        speaker_name: str,
        speaker_pos: tuple[float, float, int],
        speaker_volume: float,
        raw_utterance: str,
        grid_w: float = 256.0,
        grid_h: float = 256.0,
    ) -> bool:
        """Strictly non-blocking (<10 us) submission into background System-2 observation queue."""
        if not self.is_ready or self._engine is None:
            return False
        payload = {
            "w_state": w_state.detach(),
            "z_world": z_world.detach(),
            "bot": dict(bot),
            "speaker_name": speaker_name,
            "speaker_pos": speaker_pos,
            "speaker_volume": float(speaker_volume),
            "raw_utterance": raw_utterance,
            "grid_w": grid_w,
            "grid_h": grid_h,
        }
        try:
            self._obs_queue.put_nowait(payload)
            return True
        except queue.Full:
            try:
                self._obs_queue.get_nowait()
                self._obs_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._obs_queue.put_nowait(payload)
                return True
            except queue.Full:
                return False

    def poll_replies(self) -> list[tuple[Any, dict[str, Any]]]:
        """Strictly non-blocking (<30 us) poll of completed System-2 SLM replies + bot metadata."""
        if not self.is_ready or self._engine is None:
            return []
        resps = self._engine.poll_replies()
        if not resps:
            return []
        out: list[tuple[Any, dict[str, Any]]] = []
        with self._meta_lock:
            for r in resps:
                meta = self._req_meta.pop(r.request_id, {})
                out.append((r, meta))
        return out


class InferenceEngine:
    def __init__(
        self,
        world_pt: Path,
        eco_pt: Path,
        bot_pt: Path,
        use_fp16: bool = True,
        warmup_shapes: tuple[tuple[int, int], ...] = ((128, 128),),
        use_sparse_world: bool = True,
        use_sparse_eco: bool | None = None,
        acoustic_pt: Path | None = None,
        actuator_pt: Path | None = None,
        actuator_v3_pt: Path | None = None,
        world_3d_pt: Path | None = None,
        world_dt_pt: Path | None = None,
        world_v5_pt: Path | None = None,
        enable_system2_slm: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")
        self.lock = threading.Lock()
        if use_sparse_eco is None:
            use_sparse_eco = use_sparse_world

        if self.device.type == "cuda":
            # Disable cudnn.benchmark when sparse dynamic tile batches are enabled to avoid 1s autotune stalls
            any_sparse = bool(
                (use_sparse_world and SparseWorldRunner is not None)
                or (use_sparse_eco and SparseEcoRunner is not None)
            )
            torch.backends.cudnn.benchmark = not any_sparse
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

        self.actuator_policy: Any = None
        act_path = actuator_pt if actuator_pt is not None else (_SCRIPT_DIR / "actuator_policy_rl.pt")
        if ActuatorPolicyNet is not None and act_path.exists():
            try:
                sd_act = _load_state_dict(act_path)
                ap = ActuatorPolicyNet().eval()
                ap.load_state_dict(sd_act)
                ap.to(self.device)
                self.actuator_policy = ap
                print(f"[gpu_infer_daemon] Loaded ActuatorPolicyNet from {act_path}", flush=True)
            except Exception as exc:
                print(
                    f"[gpu_infer_daemon] ActuatorPolicyNet load failed ({exc}); using BotPolicy->Actuator fallback",
                    file=sys.stderr,
                    flush=True,
                )
                self.actuator_policy = None

        self.actuator_policy_v3: Any = None
        act_v3_path = actuator_v3_pt if actuator_v3_pt is not None else (_SCRIPT_DIR / "actuator_policy_v3_rl.pt")
        if ActuatorPolicyNetV3 is not None and act_v3_path.exists():
            try:
                sd_v3 = _load_state_dict(act_v3_path)
                ap3 = ActuatorPolicyNetV3().eval()
                ap3.load_state_dict(sd_v3)
                ap3.to(self.device)
                self.actuator_policy_v3 = ap3
                print(f"[gpu_infer_daemon] Loaded True-3D ActuatorPolicyNetV3 from {act_v3_path}", flush=True)
            except Exception as exc:
                print(
                    f"[gpu_infer_daemon] ActuatorPolicyNetV3 load failed ({exc}); using Actuator v2 fallback",
                    file=sys.stderr,
                    flush=True,
                )
                self.actuator_policy_v3 = None

        self.use_sparse_world: bool = bool(use_sparse_world and SparseWorldRunner is not None)
        self._sparse_runners: dict[tuple[int, int, int], _SparseWorldAdapter] = {}
        self.use_sparse_eco: bool = bool(use_sparse_eco and SparseEcoRunner is not None)
        self._sparse_eco_runners: dict[tuple[int, int, int], _SparseEcoAdapter] = {}

        # Optional WorldModel3D v4 (`op=8` true 3D horizontal + vertical neural physics)
        self.world_model_3d: Any = None
        self._sparse_3d_runner_cls: Any = None
        self._sparse_3d_runners: dict[tuple[int, int, int], _SparseWorld3DAdapter] = {}
        if world_3d_pt is not None and world_3d_pt.exists():
            try:
                from world_model_3d import WorldModel3D  # noqa: E402
                from sparse_world_3d_runner import SparseWorld3DRunner  # noqa: E402

                wm3d = WorldModel3D(in_ch=18, out_ch=17, width=48, num_res=5, z_flux_ch=Z_FLUX_CH).eval()
                wm3d.load_state_dict(_load_state_dict(world_3d_pt))
                wm3d.to(self.device)
                self.world_model_3d = wm3d
                self._sparse_3d_runner_cls = SparseWorld3DRunner
                if self.device.type == "cuda" and use_sparse_world:
                    torch.backends.cudnn.benchmark = False
                print(f"[gpu_infer_daemon] Loaded WorldModel3D v4 from {world_3d_pt}", flush=True)
            except Exception as exc:
                print(
                    f"[gpu_infer_daemon] WorldModel3D v4 load failed ({exc}); op=8 disabled",
                    file=sys.stderr,
                    flush=True,
                )
                self.world_model_3d = None
                self._sparse_3d_runner_cls = None

        # Optional WorldModelDT / WorldModel3D-v5 (`op=9` continuous per-voxel local_dt + 16x16x8 3D sparse bricks!)
        self.world_model_dt: Any = None
        self._sparse_dt_runner_cls: Any = None
        self._sparse_dt_runners: dict[tuple[int, int, int], _SparseWorldDTAdapter] = {}
        self.world_model_v5_active: bool = False
        dt_candidate = world_v5_pt if (world_v5_pt is not None and world_v5_pt.exists()) else world_dt_pt
        if dt_candidate is not None and dt_candidate.exists():
            try:
                sd_dt = _load_state_dict(dt_candidate)
                is_v8_cand = "v8" in dt_candidate.name.lower() or any(
                    k.startswith(("law_", "world_law", "film_")) for k in sd_dt
                )
                is_v5 = any(k.startswith("blocks.4.conv3d.") or k.startswith("h_lat_in.") for k in sd_dt)
                loaded_v8 = False
                if is_v8_cand:
                    try:
                        from world_model_3d_v5 import SparseWorld3DV5Runner  # noqa: E402
                        from world_model_3d_v8 import WorldModel3DV8  # noqa: E402

                        try:
                            from world_model_3d_v8 import SparseWorld3DV8Runner as _V8Runner  # noqa: E402
                        except Exception:
                            _V8Runner = SparseWorld3DV5Runner

                        wmv8 = _call_with_accepted_kwargs(
                            WorldModel3DV8,
                            in_ch=18,
                            out_ch=17,
                            width=48,
                            z_flux_ch=Z_FLUX_CH,
                        ).eval()
                        wmv8.load_state_dict(sd_dt, strict=False)
                        wmv8.to(self.device)
                        self.world_model_dt = wmv8
                        self._sparse_dt_runner_cls = _V8Runner
                        self.world_model_v5_active = True
                        # Also back op=8 with WorldModel3DV8 when loaded
                        self.world_model_3d = wmv8
                        self._sparse_3d_runner_cls = _V8Runner
                        if self.device.type == "cuda" and use_sparse_world:
                            torch.backends.cudnn.benchmark = False
                        print(
                            f"[gpu_infer_daemon] Loaded WorldModel3D-v8 (16x16x8 3D sparse!) from {dt_candidate}",
                            flush=True,
                        )
                        loaded_v8 = True
                    except Exception as exc_v8:
                        print(
                            f"[gpu_infer_daemon] WorldModel3D-v8 load fallback ({exc_v8}); trying WorldModel3DV5",
                            file=sys.stderr,
                            flush=True,
                        )
                if loaded_v8:
                    pass
                elif is_v5 or is_v8_cand:
                    from world_model_3d_v5 import SparseWorld3DV5Runner, WorldModel3DV5  # noqa: E402

                    wmv5 = WorldModel3DV5(in_ch=18, out_ch=17, width=48, z_flux_ch=Z_FLUX_CH).eval()
                    wmv5.load_state_dict(sd_dt, strict=not is_v8_cand)
                    wmv5.to(self.device)
                    self.world_model_dt = wmv5
                    self._sparse_dt_runner_cls = SparseWorld3DV5Runner
                    self.world_model_v5_active = True
                    # Also back op=8 with WorldModel3DV5 when loaded
                    self.world_model_3d = wmv5
                    self._sparse_3d_runner_cls = SparseWorld3DV5Runner
                    if self.device.type == "cuda" and use_sparse_world:
                        torch.backends.cudnn.benchmark = False
                    print(f"[gpu_infer_daemon] Loaded Conservation-Distilled WorldModel3D-v5 (16x16x8 3D sparse!) from {dt_candidate}", flush=True)
                else:
                    from world_model_dt import WorldModelDT  # noqa: E402
                    from sparse_world_3d_runner import SparseWorldDTRunner  # noqa: E402

                    wmdt = WorldModelDT(in_ch=18, out_ch=17, width=48, num_res=5, z_flux_ch=Z_FLUX_CH).eval()
                    wmdt.load_state_dict(sd_dt)
                    wmdt.to(self.device)
                    self.world_model_dt = wmdt
                    self._sparse_dt_runner_cls = SparseWorldDTRunner
                    if self.device.type == "cuda" and use_sparse_world:
                        torch.backends.cudnn.benchmark = False
                    print(f"[gpu_infer_daemon] Loaded WorldModelDT from {dt_candidate}", flush=True)
            except Exception as exc:
                print(
                    f"[gpu_infer_daemon] WorldModelDT/v5 load failed ({exc}); op=9 disabled",
                    file=sys.stderr,
                    flush=True,
                )
                self.world_model_dt = None
                self._sparse_dt_runner_cls = None

        # Optional non-blocking System-2 Qwen2.5-1.5B Dialogue Engine (`--system2-slm`)
        self.system2_bridge: _AsyncSystem2Bridge | None = None
        if enable_system2_slm:
            self.system2_bridge = _AsyncSystem2Bridge(device=self.device, use_fp16=self.use_fp16)

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
        cache_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.last_z_world: torch.Tensor = torch.zeros((4, 48, 256, 256), dtype=cache_dtype, device=self.device)
        self.last_z_eco: torch.Tensor = torch.zeros((1, 32, 256, 256), dtype=cache_dtype, device=self.device)
        self.last_world_state: torch.Tensor | None = None
        self._pooled_z_world: torch.Tensor | None = None
        self._pooled_z_world_src: torch.Tensor | None = None
        self._pooled_world_state: torch.Tensor | None = None
        self._pooled_world_state_src: torch.Tensor | None = None
        self.last_sparse_active_frac: float = 1.0
        self.last_sparse_eco_active_frac: float = 1.0
        self._acoustic_step_counter: int = 0
        self._last_auto_callout_ts: float = 0.0
        self._fallback_recent_lines: deque[str] = deque(maxlen=45)
        self._fallback_speaker_last: dict[int, int] = {}

        for (h, w) in warmup_shapes:
            for op in (OP_WORLD_STEP, OP_ECO_STEP):
                self._get_runner(op, h, w, 1)

    def _invalidate_strata_pool_cache(self) -> None:
        self._pooled_z_world = None
        self._pooled_z_world_src = None
        self._pooled_world_state = None
        self._pooled_world_state_src = None

    def _get_z_world_4strata(self) -> torch.Tensor:
        """Return canonical 4-strata Z_world [<=4, 48, H, W], pooling L>4 (e.g. L=16) on demand."""
        z = self.last_z_world
        if z.shape[0] <= 4:
            return z
        if self._pooled_z_world is not None and self._pooled_z_world_src is z:
            return self._pooled_z_world
        pooled = _pool_z_to_4_strata(z, is_world_state=False)
        self._pooled_z_world = pooled
        self._pooled_z_world_src = z
        return pooled

    def _get_world_state_4strata(self) -> torch.Tensor | None:
        """Return canonical 4-strata w_state [<=4, 18, H, W], pooling L>4 (e.g. L=16) on demand."""
        w = self.last_world_state
        if w is None or w.shape[0] <= 4:
            return w
        if self._pooled_world_state is not None and self._pooled_world_state_src is w:
            return self._pooled_world_state
        pooled = _pool_z_to_4_strata(w, is_world_state=True)
        self._pooled_world_state = pooled
        self._pooled_world_state_src = w
        return pooled

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

    def _get_sparse_3d_runner(self, h: int, w: int, batch: int) -> _SparseWorld3DAdapter:
        assert self.world_model_3d is not None and self._sparse_3d_runner_cls is not None
        key = (batch, h, w)
        adapter = self._sparse_3d_runners.get(key)
        if adapter is None:
            adapter = _SparseWorld3DAdapter(
                runner_cls=self._sparse_3d_runner_cls,
                model_3d=self.world_model_3d,
                h=h,
                w=w,
                batch=batch,
                device=self.device,
                use_fp16=self.use_fp16,
                use_sparse=self.use_sparse_world,
            )
            self._sparse_3d_runners[key] = adapter
            print(
                f"[gpu_infer_daemon] Activated SparseWorld3DRunner for {batch}x18x{h}x{w} (sparse={self.use_sparse_world})",
                flush=True,
            )
        return adapter

    def infer_world_3d_bytes(self, h: int, w: int, recv_fn, batch: int = 4) -> tuple[memoryview, int]:
        if self.world_model_3d is None or self._sparse_3d_runner_cls is None:
            raise RuntimeError("world_model_3d_v4.pt not loaded")
        with self.lock:
            adapter = self._get_sparse_3d_runner(h, w, batch)
            if not recv_fn(memoryview(adapter.host_in_u8)):
                raise ConnectionError("EOF reading op=8 request payload")
            out_view, z_latent, x_dev, active_frac, infer_us = adapter.infer_from_host_in()
            self.last_z_world = z_latent
            self.last_world_state = x_dev
            self._invalidate_strata_pool_cache()
            self.last_sparse_active_frac = active_frac
            out_copy = bytes(out_view)
        return memoryview(out_copy), infer_us

    def _get_sparse_dt_runner(self, h: int, w: int, batch: int) -> _SparseWorldDTAdapter:
        assert self.world_model_dt is not None and self._sparse_dt_runner_cls is not None
        key = (batch, h, w)
        adapter = self._sparse_dt_runners.get(key)
        if adapter is None:
            adapter = _SparseWorldDTAdapter(
                runner_cls=self._sparse_dt_runner_cls,
                model_dt=self.world_model_dt,
                h=h,
                w=w,
                batch=batch,
                device=self.device,
                use_fp16=self.use_fp16,
                use_sparse=self.use_sparse_world,
            )
            self._sparse_dt_runners[key] = adapter
            print(
                f"[gpu_infer_daemon] Activated SparseWorldDTRunner for {batch}x18x{h}x{w} + local_dt (sparse={self.use_sparse_world})",
                flush=True,
            )
        return adapter

    def infer_world_dt_bytes(self, h: int, w: int, recv_fn, batch: int = 4) -> tuple[memoryview, int]:
        if self.world_model_dt is None or self._sparse_dt_runner_cls is None:
            raise RuntimeError("world_model_dt.pt not loaded")
        with self.lock:
            adapter = self._get_sparse_dt_runner(h, w, batch)
            if not recv_fn(memoryview(adapter.host_in_u8)):
                raise ConnectionError("EOF reading op=9 request payload")
            out_view, z_latent, x_dev, active_frac, infer_us = adapter.infer_from_host_in()
            self.last_z_world = z_latent
            self.last_world_state = x_dev
            self._invalidate_strata_pool_cache()
            self.last_sparse_active_frac = active_frac
            out_copy = bytes(out_view)
        return memoryview(out_copy), infer_us

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

    def _get_sparse_eco_runner(self, h: int, w: int, batch: int) -> _SparseEcoAdapter | None:
        if not self.use_sparse_eco or SparseEcoRunner is None:
            return None
        key = (batch, h, w)
        adapter = self._sparse_eco_runners.get(key)
        if adapter is None:
            try:
                adapter = _SparseEcoAdapter(
                    model=self.eco_model,
                    h=h,
                    w=w,
                    batch=batch,
                    device=self.device,
                    use_fp16=self.use_fp16,
                )
                self._sparse_eco_runners[key] = adapter
                print(
                    f"[gpu_infer_daemon] Activated SparseEcoRunner for {batch}x8x{h}x{w}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[gpu_infer_daemon] SparseEcoRunner unavailable for {batch}x8x{h}x{w} ({exc}); using dense CUDA Graph",
                    file=sys.stderr,
                    flush=True,
                )
                self.use_sparse_eco = False
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
                        self._invalidate_strata_pool_cache()
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
                        self._invalidate_strata_pool_cache()
                        self.last_sparse_active_frac = 1.0
                        return out_ch, memoryview(bytes(out_view)), infer_us

            if op == OP_ECO_STEP and self.use_sparse_eco:
                e_adapter = self._get_sparse_eco_runner(h, w, batch)
                if e_adapter is not None:
                    if not recv_fn(memoryview(e_adapter.host_in_u8)):
                        raise ConnectionError("EOF reading request payload")
                    try:
                        out_view, z_latent, _x_dev, active_frac, infer_us = e_adapter.infer_from_host_in()
                        self.last_z_eco = z_latent
                        self.last_sparse_eco_active_frac = active_frac
                        return out_ch, memoryview(bytes(out_view)), infer_us
                    except Exception as exc:
                        print(
                            f"[gpu_infer_daemon] SparseEcoRunner step failed ({exc}); falling back to dense",
                            file=sys.stderr,
                            flush=True,
                        )
                        self.use_sparse_eco = False
                        runner = self._get_runner(op, h, w, batch)
                        runner.host_in.copy_(e_adapter.host_in)
                        out_view, z_latent, _x_dev, infer_us = runner.infer_from_host_in()
                        self.last_z_eco = z_latent
                        self.last_sparse_eco_active_frac = 1.0
                        return out_ch, memoryview(bytes(out_view)), infer_us

            runner = self._get_runner(op, h, w, batch)
            if not recv_fn(memoryview(runner.host_in_u8)):
                raise ConnectionError("EOF reading request payload")
            out_view, z_latent, x_dev, infer_us = runner.infer_from_host_in()
            if op == OP_WORLD_STEP:
                self.last_z_world = z_latent
                self.last_world_state = x_dev
                self._invalidate_strata_pool_cache()
                self.last_sparse_active_frac = 1.0
            elif op == OP_ECO_STEP:
                self.last_z_eco = z_latent
                self.last_sparse_eco_active_frac = 1.0
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
        layers_z = int(ws_cpu.shape[0]) if (ws_cpu is not None and ws_cpu.ndim == 4) else 4
        z_scale = 6.0 * (3.0 / float(max(3, layers_z - 1)))
        dist = math.sqrt((sx - lx) ** 2 + (sy - ly) ** 2 + ((sz - lz) * z_scale) ** 2)
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

        # Periodic emergent tactical callout from an active bot (~every 1.8s) with per-speaker cooldown & dedupe
        now = time.monotonic()
        self._acoustic_step_counter += 1
        if bots and not bubbles and (now - self._last_auto_callout_ts) >= 1.8:
            self._last_auto_callout_ts = now
            # Pick speaker not on cooldown
            picked_bot = bots[self._acoustic_step_counter % len(bots)]
            for off in range(len(bots)):
                cand = bots[(self._acoustic_step_counter + off) % len(bots)]
                cid = int(cand.get("id", off))
                if self._acoustic_step_counter - self._fallback_speaker_last.get(cid, -1000) >= 6:
                    picked_bot = cand
                    break
            b = picked_bot
            bid_sel = int(b.get("id", 0))
            self._fallback_speaker_last[bid_sel] = self._acoustic_step_counter
            bx = float(b.get("x", gw * 0.5))
            by = float(b.get("y", gh * 0.5))
            bz = float(b.get("z", 1.0))
            cls_id = int(b.get("class_id", b.get("cls", 0))) % 5
            snr_l = self._compute_physical_snr_db(bx, by, bz, lx, ly, lz, 0.82, ws_cpu)
            recent_events = req.get("recent_events") or []
            epoch_s = str(req.get("epoch", "")).strip()
            arch_lines = {
                0: ["SHIELD WALL LOCKED — HOLD VANGUARD!", "CARVING CRYO COVER FOR SQUAD!"],
                1: [f"RIDING UPDRAFT AT +{int(round(bz))}M — OIL PASS!", "AERIAL OVERWATCH ON THEIR VANGUARD!"],
                2: ["QUENCHING BURNING ALLY — PUSH THE STEAM!", "HYDRO SURGE THROUGH THE CHANNEL!"],
                3: ["ISOLATED TARGET — CLOSING IN SMOKE VEIL!", "SOLO FLANK ABOVE EAST RIDGE!"],
                4: ["CARVED UNDER SHRINE — ERUPTING BEHIND SHIELDS!", "SEISMIC TUNNEL BREACH INCOMING!"],
            }
            cand_msgs: list[tuple[int, str]] = []
            for ev in recent_events:
                lbl = str(ev.get("label", ev) if isinstance(ev, dict) else ev).strip()
                if lbl:
                    cand_msgs.append((cls_id + 1, f"{lbl} — PRESS ADVANTAGE!"))
            if epoch_s:
                cand_msgs.append((5, f"TERRAIN SHIFTING — {epoch_s}!"))
            for m in arch_lines.get(cls_id, arch_lines[0]):
                cand_msgs.append((cls_id + 1, m))
            sym_id, msg = cand_msgs[0]
            for s_i, m_txt in cand_msgs:
                if m_txt not in self._fallback_recent_lines:
                    sym_id, msg = s_i, m_txt
                    break
            if snr_l < -6.0:
                msg = "[inaudible — drowned out by terrain]"
            elif snr_l < 0.0:
                msg = "...muffled callout through static..."
            self._fallback_recent_lines.append(msg)
            bubbles.append(
                {
                    "speaker_id": bid_sel,
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
                    w_state_4 = self._get_world_state_4strata()
                    z_world_4 = self._get_z_world_4strata()
                    assert w_state_4 is not None
                    num_layers = int(self.last_world_state.shape[0])
                    raw_bots = req.get("bots") or []
                    bot_states = []
                    bot_pos_map: dict[int, tuple[float, float, float]] = {}
                    for b in raw_bots:
                        bid = int(b.get("id", 0))
                        bx = float(b.get("x", 128.0))
                        by = float(b.get("y", 128.0))
                        bz = float(b.get("z", 1.0))
                        bot_pos_map[bid] = (bx, by, bz)
                        band_z = _remap_z_to_4_strata(bz, num_layers)
                        bot_states.append(
                            {
                                "id": bid,
                                "name": str(b.get("name", f"BOT-{bid}")),
                                "team": int(b.get("team", 0)),
                                "cls": int(b.get("class_id", b.get("cls", 0))),
                                "class_id": int(b.get("class_id", b.get("cls", 0))),
                                "bot_class": b.get("bot_class", int(b.get("class_id", b.get("cls", 0)))),
                                "squad_id": int(b.get("squad_id", 0)),
                                "x": bx,
                                "y": by,
                                "z": bz,
                                "bot_z": float(b.get("bot_z", bz)),
                                "layer": band_z if num_layers > 4 else int(round(bz)),
                                "hp": float(b.get("hp_frac", b.get("hp", 1.0))),
                                "bot_hp": float(b.get("bot_hp", b.get("hp_frac", b.get("hp", 1.0)))),
                                "in_tunnel": bool(b.get("in_tunnel", False)),
                                "in_water": bool(b.get("in_water", False)),
                                "airborne": bool(b.get("airborne", False)),
                                "shield_wall_active": bool(b.get("shield_wall_active", False)),
                                "burning": bool(b.get("burning", False)),
                            }
                        )
                    lis = req.get("listener") or [128.0, 128.0, 1.0]
                    listener_pos_orig = (
                        float(lis[0]) if len(lis) > 0 else 128.0,
                        float(lis[1]) if len(lis) > 1 else 128.0,
                        float(lis[2]) if len(lis) > 2 else 1.0,
                    )
                    listener_pos_4 = (
                        listener_pos_orig[0],
                        listener_pos_orig[1],
                        _remap_z_to_4_strata(listener_pos_orig[2], num_layers),
                    )
                    chat = req.get("chat")
                    chat_4: dict[str, Any] | None = None
                    if isinstance(chat, dict) and chat.get("text"):
                        chat_4 = dict(chat)
                        if num_layers > 4 and "layer" in chat_4:
                            chat_4["layer"] = _remap_z_to_4_strata(float(chat_4["layer"]), num_layers)
                    world_ctx = {
                        "recent_events": req.get("recent_events") or [],
                        "epoch": str(req.get("epoch", "")),
                        "biome": str(req.get("biome", "")),
                        "grid_w": int(req.get("grid_w", w_state_4.shape[3])),
                        "grid_h": int(req.get("grid_h", w_state_4.shape[2])),
                    }
                    raw_res = _call_with_accepted_kwargs(
                        self.acoustic_engine.step_live,
                        w_state=w_state_4,
                        z_world=z_world_4,
                        bot_states=bot_states,
                        listener_pos=listener_pos_4,
                        human_chat=chat_4,
                        world_context=world_ctx,
                    )
                    now_ts = time.perf_counter()
                    has_chat = chat_4 is not None
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
                            bx, by, bz = bot_pos_map.get(bid, listener_pos_orig)
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
                            orig_pos = bot_pos_map.get(bid, listener_pos_orig)
                            pos = sb.get("pos") or list(orig_pos)
                            bz_out = (
                                float(orig_pos[2])
                                if num_layers > 4
                                else (float(pos[2]) if len(pos) > 2 else float(orig_pos[2]))
                            )
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
                                    "z": bz_out,
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

            # Non-blocking System-2 Qwen2.5-1.5B Dialogue Hook (<10 us submit / <30 us poll)
            slm_active = bool(self.system2_bridge is not None and self.system2_bridge.is_ready)
            if slm_active and self.system2_bridge is not None:
                chat_req = req.get("chat")
                raw_bots_slm = req.get("bots") or []
                gw_slm = float(req.get("grid_w", 256.0))
                gh_slm = float(req.get("grid_h", 256.0))
                if (
                    isinstance(chat_req, dict)
                    and str(chat_req.get("text", "")).strip()
                    and self.last_world_state is not None
                    and self.last_z_world is not None
                    and raw_bots_slm
                ):
                    slm_layers = int(self.last_world_state.shape[0])
                    w_slm_4 = self._get_world_state_4strata()
                    z_slm_4 = self._get_z_world_4strata()
                    assert w_slm_4 is not None
                    cx = float(chat_req.get("x", gw_slm * 0.5))
                    cy = float(chat_req.get("y", gh_slm * 0.5))
                    cz_raw = float(chat_req.get("layer", 1.0))
                    cz_4 = int(round(_remap_z_to_4_strata(cz_raw, slm_layers)))
                    c_team = int(chat_req.get("team", 0))
                    c_vol = float(chat_req.get("volume", 1.0))
                    c_text = str(chat_req.get("text", "")).strip()
                    allies = [b for b in raw_bots_slm if int(b.get("team", 0)) == c_team] or raw_bots_slm
                    chosen = min(
                        allies,
                        key=lambda b: math.hypot(
                            float(b.get("x", gw_slm * 0.5)) - cx,
                            float(b.get("y", gh_slm * 0.5)) - cy,
                        ),
                    )
                    chosen_slm = dict(chosen)
                    orig_bot_z = float(chosen.get("z", 1.0))
                    chosen_slm["orig_z"] = orig_bot_z
                    chosen_slm["z"] = _remap_z_to_4_strata(orig_bot_z, slm_layers)
                    self.system2_bridge.submit_async(
                        w_state=w_slm_4,
                        z_world=z_slm_4,
                        bot=chosen_slm,
                        speaker_name="Commander",
                        speaker_pos=(cx, cy, cz_4),
                        speaker_volume=c_vol,
                        raw_utterance=c_text,
                        grid_w=gw_slm,
                        grid_h=gh_slm,
                    )

                for resp, meta in self.system2_bridge.poll_replies():
                    plan = resp.plan
                    reply_txt = str(plan.get("reply", "")).strip()
                    if not reply_txt:
                        continue
                    bid = int(meta.get("bot_id", 0))
                    bx = float(meta.get("x", gw_slm * 0.5))
                    by = float(meta.get("y", gh_slm * 0.5))
                    bz = float(meta.get("z", 1.0))
                    mgw = float(meta.get("grid_w", gw_slm))
                    mgh = float(meta.get("grid_h", gh_slm))
                    sector = str(plan.get("target_sector", ""))
                    sector_targets = {
                        "NW Highland": (mgw * 0.28, mgh * 0.25),
                        "NE Skybridge": (mgw * 0.55, mgh * 0.30),
                        "SW Oil Basin": (mgw * 0.28, mgh * 0.72),
                        "SE Shrine Valley": (mgw * 0.72, mgh * 0.72),
                        "L0 Cavern": (mgw * 0.50, mgh * 0.50),
                        "Central River": (mgw * 0.50, mgh * 0.50),
                    }
                    tx, ty = sector_targets.get(sector, (bx, by))
                    dx, dy = tx - bx, ty - by
                    mag = math.hypot(dx, dy)
                    mx, my = (dx / mag, dy / mag) if mag > 1.5 else (0.0, 0.0)

                    ab = plan.get("actuator_bias") or {}
                    tf = float(ab.get("thermal_flux", 0.0))
                    matter = str(ab.get("matter", "none")).lower()
                    if matter == "oil":
                        elem_ov = 2
                    elif tf < -0.25 or matter == "water":
                        elem_ov = 1
                    elif tf > 0.25:
                        elem_ov = 0
                    else:
                        elem_ov = -1

                    if matter == "smoke":
                        ab_ov = 0
                    elif matter == "water" and tf < -0.2:
                        ab_ov = 3
                    elif abs(tf) > 0.7:
                        ab_ov = 4
                    else:
                        ab_ov = 2

                    result["bubbles"].insert(
                        0,
                        {
                            "speaker_id": bid,
                            "speaker_name": str(meta.get("bot_name", resp.bot_name)),
                            "team": int(meta.get("team", 0)),
                            "x": bx,
                            "y": by,
                            "z": bz,
                            "snr_db": float(meta.get("snr_db", 12.0)),
                            "symbol_id": 16,
                            "text": reply_txt[:72],
                        },
                    )
                    if bid > 0:
                        result["overrides"].append(
                            {
                                "bot_id": bid,
                                "move_x": round(mx, 3),
                                "move_y": round(my, 3),
                                "elem": elem_ov,
                                "ability": ab_ov,
                                "snr_db": float(meta.get("snr_db", 12.0)),
                                "ttl_ticks": 90,
                            }
                        )

            result["acoustic_ms"] = round(float(result.get("acoustic_ms", infer_us / 1000.0)), 3)
            result["sparse_pct"] = round(self.last_sparse_active_frac * 100.0, 2)
            result["system2_slm_active"] = slm_active
        out_bytes = json.dumps(result, separators=(",", ":")).encode("utf-8")
        return out_bytes, infer_us

    def infer_bots(self, n_bots: int, raw_bytes: bytes, return_saliency: bool = False) -> tuple[bytes, int]:
        if self.bot_policy is None:
            raise RuntimeError("bot_policy.pt not loaded")
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(n_bots, BOT_FEAT_DIM).copy()
        with self.lock:
            t0 = time.perf_counter()
            with torch.inference_mode():
                num_layers = int(self.last_z_world.shape[0])
                if num_layers > 4 and np.any(arr[:, 2] > 1.0):
                    # Remap raw z in [0..L-1] into canonical normalized strata [0..1] (= band_z / 3.0)
                    arr[:, 2] = np.clip(arr[:, 2] / float(num_layers - 1), 0.0, 1.0)
                z_world_4 = self._get_z_world_4strata()
                bf = torch.from_numpy(arr).to(self.device, non_blocking=False)
                if self.use_fp16:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        res = self.bot_policy(
                            z_world_4, self.last_z_eco, bf, return_saliency=return_saliency
                        )
                else:
                    res = self.bot_policy(
                        z_world_4, self.last_z_eco, bf, return_saliency=return_saliency
                    )
                if return_saliency:
                    actions, saliency = res
                    out_t = torch.cat([actions.float(), saliency.float()], dim=-1)
                else:
                    out_t = res.float()
                out_np = out_t.cpu().numpy().astype(np.float32, copy=False)
            infer_us = int((time.perf_counter() - t0) * 1_000_000)
        return out_np.tobytes(), infer_us

    def infer_actuators_v3(self, n_bots: int, raw_bytes: bytes) -> tuple[bytes, int]:
        """True 3D Volumetric ActuatorPolicyNetV3 (`op=10`): zero-copy over native Z_world_3d [L, 48, H, W]."""
        if self.actuator_policy_v3 is None or adapt_bot_feats_168 is None:
            raise RuntimeError("actuator_policy_v3_rl.pt not loaded")
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(n_bots, BOT_FEAT_DIM).copy()
        with self.lock:
            t0 = time.perf_counter()
            with torch.inference_mode():
                z_w = self.last_z_world  # native [L, 48, H, W] (L=16 or L=32, zero pooling!)
                L, _, H, W = z_w.shape
                bf = torch.from_numpy(arr).to(self.device, non_blocking=False)
                pos_xyz, self_feat, tokens, token_mask = adapt_bot_feats_168(bf, W, H, L)
                if self.use_fp16:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.actuator_policy_v3(
                            z_w,
                            pos_xyz,
                            self_feat,
                            tokens,
                            token_mask=token_mask,
                            z_eco=self.last_z_eco,
                            return_saliency=True,
                        )
                else:
                    out = self.actuator_policy_v3(
                        z_w,
                        pos_xyz,
                        self_feat,
                        tokens,
                        token_mask=token_mask,
                        z_eco=self.last_z_eco,
                        return_saliency=True,
                    )
                out_t = torch.cat([out["mean"].float(), out["saliency_4x4"].float()], dim=-1)  # [N, 28]
                out_np = out_t.cpu().numpy().astype(np.float32, copy=False)
            infer_us = int((time.perf_counter() - t0) * 1_000_000)
        return out_np.tobytes(), infer_us

    def infer_actuators(self, n_bots: int, raw_bytes: bytes) -> tuple[bytes, int]:
        if self.actuator_policy_v3 is None and self.actuator_policy is None and self.bot_policy is None:
            raise RuntimeError("neither actuator_policy_v3_rl.pt, actuator_policy_rl.pt nor bot_policy.pt loaded")
        # Transparently use True-3D ActuatorPolicyNetV3 when loaded on native Z>=8 worlds!
        if self.actuator_policy_v3 is not None and adapt_bot_feats_168 is not None and self.last_z_world.shape[0] >= 8:
            raw_v3, infer_us = self.infer_actuators_v3(n_bots, raw_bytes)
            arr_v3 = np.frombuffer(raw_v3, dtype=np.float32).reshape(n_bots, ACTUATOR_V3_INSPECT_OUT_DIM)
            # Project 12D 3D actuators -> 10D op=7 wire format + preserve exact 16D saliency
            out_10 = np.empty((n_bots, ACTUATOR_INSPECT_OUT_DIM), dtype=np.float32)
            out_10[:, 0] = arr_v3[:, 0]                    # move_x
            out_10[:, 1] = arr_v3[:, 1]                    # move_y
            out_10[:, 2] = arr_v3[:, 3]                    # aim_dx
            out_10[:, 3] = arr_v3[:, 4]                    # aim_dy
            out_10[:, 4] = arr_v3[:, 6]                    # thermal_flux
            out_10[:, 5] = arr_v3[:, 7]                    # emit_water
            out_10[:, 6] = arr_v3[:, 8]                    # emit_oil
            out_10[:, 7] = arr_v3[:, 9]                    # emit_smoke
            out_10[:, 8] = arr_v3[:, 10]                   # focus_radius
            out_10[:, 9] = np.clip(arr_v3[:, 11], 0.0, 1.0)  # kinetic impulse
            out_10[:, 10:26] = arr_v3[:, 12:28]            # 4x4 saliency
            return out_10.tobytes(), infer_us
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(n_bots, BOT_FEAT_DIM).copy()
        with self.lock:
            t0 = time.perf_counter()
            with torch.inference_mode():
                num_layers = int(self.last_z_world.shape[0])
                if num_layers > 4 and np.any(arr[:, 2] > 1.0):
                    # Remap raw z in [0..L-1] into canonical normalized strata [0..1] (= band_z / 3.0)
                    arr[:, 2] = np.clip(arr[:, 2] / float(num_layers - 1), 0.0, 1.0)
                z_world_4 = self._get_z_world_4strata()
                bf = torch.from_numpy(arr).to(self.device, non_blocking=False)
                if self.actuator_policy is not None:
                    if self.use_fp16:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            res = self.actuator_policy(
                                z_world_4, self.last_z_eco, bf, return_saliency=True
                            )
                    else:
                        res = self.actuator_policy(
                            z_world_4, self.last_z_eco, bf, return_saliency=True
                        )
                    if isinstance(res, tuple):
                        act, saliency = res
                        out_t = torch.cat([act.float(), saliency.float()], dim=-1)
                    else:
                        out_t = res.float()
                else:
                    # Clean fallback: run BotPolicyNetV2/BotPolicyNet and map to 10 continuous physical actuators + 16 saliency
                    if self.use_fp16:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            actions, saliency = self.bot_policy(
                                z_world_4, self.last_z_eco, bf, return_saliency=True
                            )
                    else:
                        actions, saliency = self.bot_policy(
                            z_world_4, self.last_z_eco, bf, return_saliency=True
                        )
                    a = actions.float()
                    sal = saliency.float()
                    move_x = a[:, 0]
                    move_y = a[:, 1]
                    aim_dx = a[:, 2]
                    aim_dy = a[:, 3]
                    p_dash = a[:, 4]
                    p_melee = a[:, 5]
                    p_bolt = a[:, 6]
                    p_carve = a[:, 7]
                    p_e = a[:, 8]
                    p_r = a[:, 9]
                    p_fire = a[:, 10]
                    p_frost = a[:, 11]
                    p_oil = a[:, 12]

                    cast_gate = F.relu(torch.clamp(p_bolt + 0.65 * p_e + 0.90 * p_r, 0.0, 1.0) - 0.30) / 0.70
                    thermal_flux = torch.clamp((p_fire - p_frost) * cast_gate, -1.0, 1.0)
                    sp_water = torch.clamp(0.35 * p_carve * p_frost + 0.25 * p_e * p_frost, 0.0, 1.0)
                    sp_oil = torch.clamp(p_oil * cast_gate, 0.0, 1.0)
                    sp_smoke = torch.clamp(0.55 * p_dash + 0.35 * p_e * (1.0 - p_frost), 0.0, 1.0)
                    focus_radius = torch.clamp(0.20 + 0.50 * p_e + 0.80 * p_r, 0.0, 1.0)
                    impulse = torch.clamp(0.75 * p_melee + 0.50 * p_dash + 0.85 * p_r, 0.0, 1.0)

                    act10 = torch.stack(
                        [
                            move_x,
                            move_y,
                            aim_dx,
                            aim_dy,
                            thermal_flux,
                            sp_water,
                            sp_oil,
                            sp_smoke,
                            focus_radius,
                            impulse,
                        ],
                        dim=-1,
                    )
                    out_t = torch.cat([act10, sal], dim=-1)
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

                if op == OP_WORLD_DT_STEP:
                    batch = max(1, int(pad_batch))
                    expected_bytes = (batch * 18 + 1) * height * width * 4
                    if (
                        self.engine.world_model_dt is None
                        or channels != 18
                        or height == 0
                        or width == 0
                        or payload_bytes != expected_bytes
                    ):
                        if 0 < payload_bytes <= 512 * 1024 * 1024:
                            _recv_exact(conn, memoryview(bytearray(payload_bytes)))
                        self._send_error(conn, STATUS_BAD_REQUEST, height, width)
                        continue
                    try:
                        out_bytes, infer_us = self.engine.infer_world_dt_bytes(
                            height, width, lambda mv: _recv_exact(conn, mv), batch=batch
                        )
                    except Exception as exc:
                        print(
                            f"[gpu_infer_daemon] WorldModelDT op=9 error ({height}x{width}): {exc}",
                            file=sys.stderr,
                        )
                        self._send_error(conn, STATUS_INTERNAL_ERROR, height, width)
                        continue
                    resp_hdr = HEADER_STRUCT.pack(
                        STATUS_OK, batch, 17, height, width, len(out_bytes), infer_us
                    )
                    conn.sendall(resp_hdr)
                    conn.sendall(out_bytes)
                    continue

                if op == OP_WORLD_3D_STEP:
                    batch = max(1, int(pad_batch))
                    expected_bytes = batch * 18 * height * width * 4
                    if (
                        self.engine.world_model_3d is None
                        or channels != 18
                        or height == 0
                        or width == 0
                        or payload_bytes != expected_bytes
                    ):
                        if 0 < payload_bytes <= 512 * 1024 * 1024:
                            _recv_exact(conn, memoryview(bytearray(payload_bytes)))
                        self._send_error(conn, STATUS_BAD_REQUEST, height, width)
                        continue
                    try:
                        out_bytes, infer_us = self.engine.infer_world_3d_bytes(
                            height, width, lambda mv: _recv_exact(conn, mv), batch=batch
                        )
                    except Exception as exc:
                        print(
                            f"[gpu_infer_daemon] WorldModel3D op=8 error ({height}x{width}): {exc}",
                            file=sys.stderr,
                        )
                        self._send_error(conn, STATUS_INTERNAL_ERROR, height, width)
                        continue
                    resp_hdr = HEADER_STRUCT.pack(
                        STATUS_OK, batch, 17, height, width, len(out_bytes), infer_us
                    )
                    conn.sendall(resp_hdr)
                    conn.sendall(out_bytes)
                    continue

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

                if op == OP_ACTUATOR_V3:
                    n_bots = int(height)
                    expected_bytes = n_bots * BOT_FEAT_DIM * 4
                    if (
                        self.engine.actuator_policy_v3 is None
                        or n_bots <= 0
                        or width != BOT_FEAT_DIM
                        or payload_bytes != expected_bytes
                    ):
                        if 0 < payload_bytes <= 16 * 1024 * 1024:
                            _recv_exact(conn, memoryview(bytearray(payload_bytes)))
                        self._send_error(conn, STATUS_BAD_REQUEST, height, width)
                        continue
                    raw_in = bytearray(payload_bytes)
                    if not _recv_exact(conn, memoryview(raw_in)):
                        break
                    try:
                        out_bytes, infer_us = self.engine.infer_actuators_v3(n_bots, bytes(raw_in))
                    except Exception as exc:
                        print(f"[gpu_infer_daemon] ActuatorV3 op=10 error: {exc}", file=sys.stderr)
                        self._send_error(conn, STATUS_INTERNAL_ERROR, height, width)
                        continue
                    resp_hdr = HEADER_STRUCT.pack(
                        STATUS_OK, 1, 1, n_bots, ACTUATOR_V3_INSPECT_OUT_DIM, len(out_bytes), infer_us
                    )
                    conn.sendall(resp_hdr)
                    conn.sendall(out_bytes)
                    continue

                if op == OP_ACTUATOR_POLICY:
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
                        out_bytes, infer_us = self.engine.infer_actuators(n_bots, bytes(raw_in))
                    except Exception as exc:
                        print(f"[gpu_infer_daemon] Actuator policy error: {exc}", file=sys.stderr)
                        self._send_error(conn, STATUS_INTERNAL_ERROR, height, width)
                        continue
                    resp_hdr = HEADER_STRUCT.pack(
                        STATUS_OK, 1, 1, n_bots, ACTUATOR_INSPECT_OUT_DIM, len(out_bytes), infer_us
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
                    if 0 < payload_bytes <= 512 * 1024 * 1024:
                        _recv_exact(conn, memoryview(bytearray(payload_bytes)))
                    self._send_error(conn, STATUS_BAD_REQUEST, height, width)
                    continue

                expected_bytes = batch * channels * height * width * 4
                if payload_bytes != expected_bytes:
                    if 0 < payload_bytes <= 512 * 1024 * 1024:
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
    parser.add_argument(
        "--world-model-3d",
        type=Path,
        default=_SCRIPT_DIR / "world_model_3d_v4.pt",
        help="Optional WorldModel3D v4 checkpoint path for op=8 true 3D vertical physics",
    )
    parser.add_argument(
        "--world-model-dt",
        type=Path,
        default=_SCRIPT_DIR / "world_model_dt.pt",
        help="Optional WorldModelDT checkpoint path for op=9 spatial time-dilation domes",
    )
    parser.add_argument(
        "--world-model-v5",
        type=Path,
        default=resolve_default_world_model_v5(_SCRIPT_DIR),
        help="Optional conservation-distilled WorldModel3D-v7/v6/v5 checkpoint (auto-prefers v7 -> v6 -> v5 and overrides op=8/op=9 with 16x16x8 3D sparse bricks!)",
    )
    parser.add_argument(
        "--system2-slm",
        action="store_true",
        default=False,
        help="Enable non-blocking background System-2 Qwen2.5-1.5B dialogue engine in op=6 (default: False)",
    )
    parser.add_argument("--eco-model", type=Path, default=_SCRIPT_DIR / "ecosystem_model.pt")
    parser.add_argument("--bot-policy", type=Path, default=_SCRIPT_DIR / "bot_policy.pt")
    parser.add_argument("--actuator-policy", type=Path, default=_SCRIPT_DIR / "actuator_policy_rl.pt")
    parser.add_argument("--actuator-policy-v3", type=Path, default=_SCRIPT_DIR / "actuator_policy_v3_rl.pt")
    parser.add_argument("--acoustic-ckpt", type=Path, default=_SCRIPT_DIR / "acoustic_comm_live.pt")
    parser.add_argument(
        "--sparse-world",
        action=argparse.BooleanOptionalAction,
        default=(SparseWorldRunner is not None),
        help="Use SparseWorldRunner active-tile acceleration for op=1 when available (default: True if importable)",
    )
    parser.add_argument(
        "--sparse-eco",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use SparseEcoRunner active-tile acceleration for op=2 (default: enabled automatically with --sparse-world)",
    )
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    use_sparse_eco = args.sparse_world if args.sparse_eco is None else args.sparse_eco
    host, port = parse_bind(args.bind)
    engine = InferenceEngine(
        args.world_model,
        args.eco_model,
        args.bot_policy,
        use_fp16=not args.fp32,
        use_sparse_world=args.sparse_world,
        use_sparse_eco=use_sparse_eco,
        acoustic_pt=args.acoustic_ckpt,
        actuator_pt=args.actuator_policy,
        actuator_v3_pt=args.actuator_policy_v3,
        world_3d_pt=args.world_model_3d,
        world_dt_pt=args.world_model_dt,
        world_v5_pt=args.world_model_v5,
        enable_system2_slm=args.system2_slm,
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
        f"fp16={engine.use_fp16} | world_3d={engine.world_model_3d is not None} | "
        f"world_dt={engine.world_model_dt is not None} | "
        f"world_v5={engine.world_model_v5_active} | "
        f"system2_slm={args.system2_slm} | "
        f"neural_bots={engine.bot_policy is not None} | "
        f"actuator_policy={engine.actuator_policy is not None} | "
        f"actuator_v3={engine.actuator_policy_v3 is not None} | "
        f"sparse_world={engine.use_sparse_world} | sparse_eco={engine.use_sparse_eco} | "
        f"acoustics={'neural' if engine.acoustic_engine is not None else 'physical-fallback'}",
        flush=True,
    )
    daemon.serve_forever()


if __name__ == "__main__":
    main()
