#!/usr/bin/env python3
"""
Frostmarsh Ultra-Fast PyTorch CUDA FP16 3D World + Ecosystem + Shared-Latent Neural Bot Daemon.

Serves low-latency GPU inference on TCP_NODELAY and optional Unix domain socket:
  - op=1: WorldModel      ([B, 18, H, W] f32 -> [B, 17, H, W] f32) + ZERO-COPY caches Z_world [B, 48, H, W] in VRAM!
  - op=2: EcosystemModel  ([1,  8, H, W] f32 -> [1,  4, H, W] f32) + ZERO-COPY caches Z_eco   [1, 32, H, W] in VRAM!
  - op=4: BotPolicyNet    ([N, 168]      f32 -> [N, 13]       f32) reads cached Z_world & Z_eco ZERO-COPY from VRAM!
  - op=5: BotPolicyInspect([N, 168]      f32 -> [N, 29]       f32) returns 13 action floats + 16 (4x4) Z_world saliency floats!
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bot_policy import ACTION_OUT_DIM, BOT_FEAT_DIM, INSPECT_OUT_DIM, BotPolicyNet  # noqa: E402
from model import EcosystemModel, WorldModel  # noqa: E402

HEADER_STRUCT = struct.Struct("<BBHHHII")
HEADER_SIZE = HEADER_STRUCT.size  # 16 bytes

OP_WORLD_STEP = 1
OP_ECO_STEP = 2
OP_BOT_POLICY = 4
OP_BOT_POLICY_INSPECT = 5

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

    def infer_from_host_in(self) -> tuple[memoryview, torch.Tensor, int]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            if self.graph is not None and self.static_out is not None and self.static_latent is not None:
                self.static_in.copy_(self.host_in, non_blocking=True)
                self.graph.replay()
                self.host_out.copy_(self.static_out, non_blocking=False)
                z = self.static_latent
            else:
                x = self.host_in.to(self.device)
                y, z = self._run_model(self._model, x)
                self.host_out.copy_(y.cpu())
        infer_us = int((time.perf_counter() - t0) * 1_000_000)
        return self.host_out_bytes, z, infer_us


class InferenceEngine:
    def __init__(
        self,
        world_pt: Path,
        eco_pt: Path,
        bot_pt: Path,
        use_fp16: bool = True,
        warmup_shapes: tuple[tuple[int, int], ...] = ((128, 128),),
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")
        self.lock = threading.Lock()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        self.world_model = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).eval()
        self.world_model.load_state_dict(_load_state_dict(world_pt))
        self.world_model.to(self.device)

        self.eco_model = EcosystemModel(in_ch=8, out_ch=4, width=32, num_res=5).eval()
        self.eco_model.load_state_dict(_load_state_dict(eco_pt))
        self.eco_model.to(self.device)

        self.bot_policy: BotPolicyNet | None = None
        if bot_pt.exists():
            bp = BotPolicyNet().eval()
            bp.load_state_dict(_load_state_dict(bot_pt))
            bp.to(self.device)
            self.bot_policy = bp
            print(f"[gpu_infer_daemon] Loaded Shared-Latent BotPolicyNet from {bot_pt}", flush=True)

        self.models = {
            OP_WORLD_STEP: self.world_model,
            OP_ECO_STEP: self.eco_model,
        }
        self._runners: dict[tuple[int, int, int, int], _CUDAGraphRunner] = {}

        # Zero-copy shared VRAM latent cache (updated automatically on every world & eco step!)
        self.last_z_world: torch.Tensor = torch.zeros((4, 48, 256, 256), dtype=torch.float16, device=self.device)
        self.last_z_eco: torch.Tensor = torch.zeros((1, 32, 256, 256), dtype=torch.float16, device=self.device)

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

    def infer_bytes(self, op: int, h: int, w: int, recv_fn, batch: int = 1) -> tuple[int, memoryview, int]:
        _, out_ch = EXPECTED_SPECS[op]
        with self.lock:
            runner = self._get_runner(op, h, w, batch)
            if not recv_fn(memoryview(runner.host_in_u8)):
                raise ConnectionError("EOF reading request payload")
            out_view, z_latent, infer_us = runner.infer_from_host_in()
            if op == OP_WORLD_STEP:
                self.last_z_world = z_latent
            elif op == OP_ECO_STEP:
                self.last_z_eco = z_latent
            out_copy = bytes(out_view)
        return out_ch, memoryview(out_copy), infer_us

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
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    host, port = parse_bind(args.bind)
    engine = InferenceEngine(args.world_model, args.eco_model, args.bot_policy, use_fp16=not args.fp32)
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
        f"fp16={engine.use_fp16} | neural_bots={engine.bot_policy is not None}",
        flush=True,
    )
    daemon.serve_forever()


if __name__ == "__main__":
    main()
