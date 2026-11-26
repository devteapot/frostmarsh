#!/usr/bin/env python3
"""
Zero-copy Python ctypes wrapper around Frostmarsh's native Rust Vectorized RL Arena
(`crates/field-core/src/rl_arena.rs` -> `libfield_core.{dylib,so}`).

Supports PufferLib-style double-buffered async stepping across arbitrary parallel arenas
and population sizes (12 / 32 / 60 / 128 bots/arena) reusing exact `field-core::Field` physics.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import time
from pathlib import Path

import numpy as np

RL_BASE_FEAT_DIM = 168
RL_BRIEF_DIM = 16
RL_OBS_DIM = 184
RL_ACTION_DIM = 13
RL_PRED_DIM = 6
RL_BRIEF_IN_DIM = 10
RL_FIELD_CH = 18
RL_CROP_CH = 8
RL_CROP_SIZE = 16

OBJ_FREE_ROAM = 0
OBJ_CAPTURE_SHRINE = 1
OBJ_ESCORT_RELIC = 2
OBJ_HUNT_ENEMY = 3
OBJ_DEFEND_HOLD = 4

STANCE_COMMIT = 0
STANCE_HOLD = 1
STANCE_FLEE_KITE = 2


def _default_lib_name() -> str:
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        return "libfield_core.dylib"
    if sys_name == "windows":
        return "field_core.dll"
    return "libfield_core.so"


def find_field_core_lib(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"libfield_core not found at {p}")
    env_p = os.environ.get("FROSTMARSH_LIB_PATH")
    if env_p and Path(env_p).exists():
        return Path(env_p)

    lib_name = _default_lib_name()
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "target" / "release" / lib_name,
        here.parent / "target" / "debug" / lib_name,
        Path("/home/gooning/.t3/scratch/frostmarsh-quic/target/release") / lib_name,
        Path.cwd() / "target" / "release" / lib_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not locate {lib_name}. Build it with: cargo build --release -p field-core"
    )


def _bind_lib(lib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(lib_path))
    c_void_p = ctypes.c_void_p
    c_f32_p = ctypes.POINTER(ctypes.c_float)
    c_u8_p = ctypes.POINTER(ctypes.c_uint8)

    lib.fm_arena_create.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint8,
    ]
    lib.fm_arena_create.restype = c_void_p

    lib.fm_arena_destroy.argtypes = [c_void_p]
    lib.fm_arena_destroy.restype = None

    lib.fm_arena_reset.argtypes = [c_void_p, ctypes.c_uint64]
    lib.fm_arena_reset.restype = None

    lib.fm_arena_set_scripted_mask.argtypes = [c_void_p, c_u8_p, ctypes.c_size_t]
    lib.fm_arena_set_scripted_mask.restype = None

    lib.fm_arena_set_briefs.argtypes = [c_void_p, c_f32_p, ctypes.c_size_t]
    lib.fm_arena_set_briefs.restype = None

    lib.fm_arena_sample_random_briefs.argtypes = [c_void_p, ctypes.c_float]
    lib.fm_arena_sample_random_briefs.restype = None

    lib.fm_arena_step.argtypes = [c_void_p, c_f32_p, ctypes.c_size_t]
    lib.fm_arena_step.restype = None

    lib.fm_arena_step_async_begin.argtypes = [c_void_p, c_f32_p, ctypes.c_size_t]
    lib.fm_arena_step_async_begin.restype = None

    lib.fm_arena_step_async_wait.argtypes = [c_void_p]
    lib.fm_arena_step_async_wait.restype = None

    lib.fm_arena_obs_ptr.argtypes = [c_void_p]
    lib.fm_arena_obs_ptr.restype = c_f32_p

    lib.fm_arena_rew_ptr.argtypes = [c_void_p]
    lib.fm_arena_rew_ptr.restype = c_f32_p

    lib.fm_arena_done_ptr.argtypes = [c_void_p]
    lib.fm_arena_done_ptr.restype = c_u8_p

    lib.fm_arena_pred_ptr.argtypes = [c_void_p]
    lib.fm_arena_pred_ptr.restype = c_f32_p

    lib.fm_arena_crop_ptr.argtypes = [c_void_p]
    lib.fm_arena_crop_ptr.restype = c_f32_p

    lib.fm_arena_export_field_tensors.argtypes = [c_void_p, c_f32_p, ctypes.c_size_t]
    lib.fm_arena_export_field_tensors.restype = ctypes.c_int32

    lib.fm_arena_get_team_scores.argtypes = [c_void_p, c_f32_p, ctypes.c_size_t]
    lib.fm_arena_get_team_scores.restype = None
    return lib


class RustVecArena:
    """Zero-copy NumPy interface to Frostmarsh's double-buffered Rust RL arena."""

    def __init__(
        self,
        num_arenas: int = 16,
        bots_per_arena: int = 60,
        grid_w: int = 128,
        grid_h: int = 128,
        seed: int = 42,
        num_threads: int = 0,
        field_step_interval: int = 4,
        enable_crops: bool = False,
        lib_path: str | Path | None = None,
    ):
        self.lib_path = find_field_core_lib(lib_path)
        self._lib = _bind_lib(self.lib_path)
        self.num_arenas = int(num_arenas)
        self.bots_per_arena = int(bots_per_arena)
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.enable_crops = bool(enable_crops)
        self.total_bots = self.num_arenas * self.bots_per_arena

        self._ptr = self._lib.fm_arena_create(
            self.num_arenas,
            self.bots_per_arena,
            self.grid_w,
            self.grid_h,
            int(seed),
            int(num_threads),
            int(field_step_interval),
            1 if self.enable_crops else 0,
        )
        if not self._ptr:
            raise RuntimeError("fm_arena_create returned null")

        self._field_buf = np.empty(
            (self.num_arenas, RL_FIELD_CH, self.grid_h, self.grid_w), dtype=np.float32
        )
        self._score_buf = np.empty((self.num_arenas, 4), dtype=np.float32)

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            self._lib.fm_arena_destroy(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.close()

    def _views(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a, n = self.num_arenas, self.bots_per_arena
        obs = np.ctypeslib.as_array(
            self._lib.fm_arena_obs_ptr(self._ptr), shape=(a, n, RL_OBS_DIM)
        )
        rew = np.ctypeslib.as_array(
            self._lib.fm_arena_rew_ptr(self._ptr), shape=(a, n)
        )
        done = np.ctypeslib.as_array(
            self._lib.fm_arena_done_ptr(self._ptr), shape=(a, n)
        )
        pred = np.ctypeslib.as_array(
            self._lib.fm_arena_pred_ptr(self._ptr), shape=(a, n, RL_PRED_DIM)
        )
        return obs, rew, done, pred

    def obs(self) -> np.ndarray:
        return self._views()[0]

    def reset(self, seed: int = 42) -> np.ndarray:
        self._lib.fm_arena_reset(self._ptr, int(seed))
        return self.obs()

    def set_scripted_mask(self, mask: np.ndarray) -> None:
        arr = np.ascontiguousarray(mask, dtype=np.uint8)
        self._lib.fm_arena_set_scripted_mask(
            self._ptr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), arr.size
        )

    def set_briefs(self, briefs: np.ndarray) -> None:
        arr = np.ascontiguousarray(briefs, dtype=np.float32)
        self._lib.fm_arena_set_briefs(
            self._ptr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size
        )

    def sample_random_briefs(self, active_prob: float = 0.75) -> None:
        self._lib.fm_arena_sample_random_briefs(self._ptr, float(active_prob))

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arr = np.ascontiguousarray(actions, dtype=np.float32)
        self._lib.fm_arena_step(
            self._ptr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size
        )
        return self._views()

    def step_async_begin(self, actions: np.ndarray) -> None:
        arr = np.ascontiguousarray(actions, dtype=np.float32)
        self._lib.fm_arena_step_async_begin(
            self._ptr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size
        )

    def step_async_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._lib.fm_arena_step_async_wait(self._ptr)
        return self._views()

    def export_field_tensors(self) -> np.ndarray:
        ok = self._lib.fm_arena_export_field_tensors(
            self._ptr,
            self._field_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._field_buf.size,
        )
        if not ok:
            raise RuntimeError("fm_arena_export_field_tensors failed")
        return self._field_buf

    def get_team_scores(self) -> np.ndarray:
        self._lib.fm_arena_get_team_scores(
            self._ptr,
            self._score_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._score_buf.size,
        )
        return self._score_buf.copy()


def run_benchmark(num_arenas: int = 16, steps: int = 200, grid: int = 64) -> None:
    print(f"Benchmarking RustVecArena ({find_field_core_lib()}) | arenas={num_arenas} | grid={grid}x{grid} | steps={steps}")
    for n_bots in (12, 32, 60):
        arena = RustVecArena(
            num_arenas=num_arenas,
            bots_per_arena=n_bots,
            grid_w=grid,
            grid_h=grid,
            seed=42,
            field_step_interval=4,
        )
        arena.set_scripted_mask(np.ones((num_arenas, n_bots), dtype=np.uint8))
        arena.sample_random_briefs(0.8)
        act = np.zeros((num_arenas, n_bots, RL_ACTION_DIM), dtype=np.float32)
        # Warmup
        for _ in range(10):
            arena.step(act)
        t0 = time.perf_counter()
        for _ in range(steps):
            obs, rew, done, pred = arena.step(act)
        dt = time.perf_counter() - t0
        sps = (num_arenas * n_bots * steps) / max(dt, 1e-9)
        assert np.isfinite(obs).all() and np.isfinite(rew).all() and np.isfinite(pred).all()
        print(
            f"  bots={n_bots:2d}/arena ({num_arenas*n_bots:4d} total) | "
            f"step={dt/steps*1000:.2f} ms | {sps:,.0f} agent-steps/sec | "
            f"mean_pred_close={pred[..., 0].mean():.3f}"
        )
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--arenas", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--grid", type=int, default=64)
    args = parser.parse_args()
    if args.bench:
        run_benchmark(num_arenas=args.arenas, steps=args.steps, grid=args.grid)
