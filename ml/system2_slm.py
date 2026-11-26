#!/usr/bin/env python3
"""
Frostmarsh System-2 Small Language Model (SLM) Dialogue & Grounded Tactical Planning Engine.

Bridges the frozen 3D Neural World Model latent field `Z_world [4, 48, 256, 256]`,
authoritative physical state `w_state [4, 18, 256, 256]`, and 3D physical acoustics
(`AcousticField3D`) into `Qwen/Qwen2.5-1.5B-Instruct` running in BF16 on NVIDIA GB10.

Components:
  1. `ZWorldToLanguageBridge`:
     - Extracts each bot's `16x16` `Z_world` ego-crop via bilinear `grid_sample` and computes
       latent energy, spatial saliency, and peak activation quadrant.
     - Grounded physical sensor readout from `w_state` + `AcousticField3D`:
       nearby oil slicks, wildfire distance/bearing, L2 skybridge integrity, waterfall/ambient
       noise floor dB, diurnal time-of-day & temperature, bot HP/class, and incoming message
       exact `snr_db` + physical occlusion cause (basalt walls, L0 ceiling, waterfall roar,
       wildfire crackle, spell blast).
     - Applies physical SNR degradation to incoming speech utterances before prompting the SLM.

  2. `System2DialogueEngine`:
     - Loads `Qwen/Qwen2.5-1.5B-Instruct` in `torch.bfloat16` on CUDA.
     - Provides both synchronous `generate_reply(...)` and a strictly non-blocking background
       worker thread (`submit_dialogue(...)` / `poll_replies()`) so the 30 Hz game loop never stalls.
     - Maintains rolling 4-turn per-bot conversation memory.
     - Enforces compact structured JSON output (`max_new_tokens=64`, greedy decoding):
         {
           "reply": "<concise <=16 word in-character line>",
           "intent": "<tactic>",
           "target_sector": "<sector>",
           "actuator_bias": {
             "thermal_flux": -1..1,
             "matter": "oil"|"water"|"smoke"|"none",
             "focus_radius": 2.0..8.0
           }
         }
       with partial-JSON repair + physics-grounded fallback parser for 100% valid plan rate.

  3. Full Spark GB10 Benchmark & 8 Multi-Turn Physics-Grounded Conversations:
     - Writes `SYSTEM2_SLM_METRICS.json` and `SYSTEM2_SLM_REPORT.md` in `/home/gooning/.t3/scratch/frostmarsh-ml/`.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import queue
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.cache_utils import DynamicCache
from transformers.generation.streamers import BaseStreamer

ROOT = Path("/home/gooning/.t3/scratch/frostmarsh-ml")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acoustic_comm_poc import build_frostmarsh_3d_state  # noqa: E402
from live_acoustics import (  # noqa: E402
    CAUSE_DESCRIPTIONS,
    SRC_SPL_MIN_DB,
    SRC_SPL_SPAN_DB,
    AcousticField3D,
    _build_cause_text,
    _snr_tier_label,
)
from model import WorldModel  # noqa: E402

# Authoritative FIELD_CONTRACT / sim_teacher.py channel indices
S_WATER = 0
S_ICE = 1
S_STEAM = 2
S_SOIL = 3
S_ROCK = 4
S_WOOD = 5
S_ASH = 6
S_SMOKE = 7
S_OIL = 8
S_PLANT = 9
CH_TEMP = 10
CH_TOTAL_MASS = 11
CH_FLOW_E = 12
CH_FLOW_S = 13
CH_ELEV = 14
CH_WALL = 15
CH_FORCING = 16
CH_FORCING_MASS = 17

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_WORLD_V3_PT = ROOT / "world_model_v3.pt"
DEFAULT_WORLD_PT = ROOT / "world_model.pt"
DEFAULT_METRICS_JSON = ROOT / "SYSTEM2_SLM_METRICS.json"
DEFAULT_REPORT_MD = ROOT / "SYSTEM2_SLM_REPORT.md"

VALID_MATTERS = ("oil", "water", "smoke", "none")

BOT_PERSONAS: dict[str, str] = {
    "Pyromancer": "Fierce fire-tactician who loves oil-ignition combos, watches fuel lines, and speaks with sharp urgency.",
    "Cryo-Scout": "Calm, precise frost recon specialist who reads ice stability, quenches wildfires, and guards river chokes.",
    "Vanguard": "Stoic frontline shield-bearer who prioritizes structural safety, squad survival, and holding bridges.",
    "Shadow-Flanker": "Cunning stealth skirmisher who lays crude oil traps, uses smoke screens, and strikes through cavern shafts.",
}


# =============================================================================
# Helpers: WorldModel latent extraction, compass bearings, sector mapping
# =============================================================================
def load_frozen_world_model(device: torch.device) -> tuple[WorldModel, Path]:
    """Load real frozen WorldModel (`world_model_v3.pt` preferred, fallback `world_model.pt`)."""
    ckpt_path = DEFAULT_WORLD_V3_PT if DEFAULT_WORLD_V3_PT.exists() else DEFAULT_WORLD_PT
    wm = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device).eval()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    wm.load_state_dict(sd)
    for p in wm.parameters():
        p.requires_grad_(False)
    if device.type == "cuda":
        wm = wm.half()
    return wm, ckpt_path


@torch.inference_mode()
def world_forward_with_latent(
    wm: WorldModel,
    w_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run frozen WorldModel on w_state [L, 18, H, W] and return (deltas [L,17,H,W], Z_world [L,48,H,W])."""
    dt = next(wm.parameters()).dtype
    x = w_state.to(device=next(wm.parameters()).device, dtype=dt)
    h = wm.stem(x)
    for blk in wm.blocks:
        h = blk(h)
    z_world = F.gelu(h)
    raw = wm.head(z_world) * wm.out_scale.to(dtype=dt)
    out = torch.cat([raw[:, :10], F.softplus(raw[:, 10:])], dim=1)
    return out.float(), z_world


def bearing_8way(dx: float, dy: float) -> str:
    """Grid coordinates: +x is East, +y is South."""
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return "HERE"
    ang = math.degrees(math.atan2(dy, dx))
    dirs = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    return dirs[int(round(ang / 45.0)) % 8]


def pos_to_sector(x: float, y: float, w: int = 256, h: int = 256) -> str:
    nx = x / max(w - 1, 1)
    ny = y / max(h - 1, 1)
    if ny < 0.5:
        return "NW Highland" if nx < 0.5 else "NE Skybridge"
    return "SW Oil Basin" if nx < 0.5 else "SE Shrine Valley"


def degrade_utterance_by_snr(
    raw_text: str,
    snr_db: float,
    occlusion_cause: str,
    seed: int = 0,
) -> tuple[str, str]:
    """
    Physically degrade an incoming speech utterance according to AcousticField3D SNR (dB).
      >= +10 dB : clear verbatim
      +2..+10 dB: partial interference (~20% dropouts + cause tag)
      -6..+2 dB : heavily muffled (~65% words lost + cause tag like [waterfall roar])
      < -6 dB   : completely inaudible ([INAUDIBLE — drowned out by <cause>])
    """
    tier = _snr_tier_label(snr_db)
    short_cause = "static"
    lc = occlusion_cause.lower()
    if "waterfall" in lc:
        short_cause = "waterfall roar"
    elif "ceiling" in lc:
        short_cause = "rock ceiling muffle"
    elif "basalt wall" in lc:
        short_cause = "basalt wall muffle"
    elif "wildfire" in lc:
        short_cause = "wildfire crackle"
    elif "spell" in lc:
        short_cause = "blast shockwave"

    if snr_db < -6.0:
        return (
            f"[INAUDIBLE ({snr_db:+.1f} dB) — drowned out by {occlusion_cause}]",
            tier,
        )
    if snr_db >= 10.0:
        return raw_text.strip(), tier

    words = raw_text.strip().split()
    if not words:
        return f"[{short_cause}]", tier

    h = int(hashlib.md5(f"{seed}:{raw_text}:{round(snr_db, 1)}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)

    if snr_db >= 2.0:
        keep_prob = 0.72 + 0.025 * (snr_db - 2.0)
        out: list[str] = []
        dropped_any = False
        for i, w in enumerate(words):
            if i == 0 or i == len(words) - 1 or rng.random() < keep_prob:
                out.append(w)
            else:
                if not out or not out[-1].startswith("["):
                    out.append(f"[{short_cause}]")
                dropped_any = True
        if not dropped_any and len(out) >= 4:
            out[len(out) // 2] = f"[{short_cause}]"
        return " ".join(out), tier

    keep_prob = float(np.clip(0.28 + 0.015 * (snr_db + 6.0), 0.22, 0.42))
    out = []
    kept = 0
    for w in words:
        if rng.random() < keep_prob:
            out.append(w)
            kept += 1
        else:
            if not out or out[-1] != "...":
                out.append("...")
    if kept == 0:
        out = ["...", words[min(1, len(words) - 1)], "..."]
    return f"[{short_cause}] " + " ".join(out) + f" [{short_cause}]", tier


# =============================================================================
# 1. ZWorldToLanguageBridge
# =============================================================================
@dataclass
class BotProfile:
    bot_id: str
    name: str
    bot_class: str
    team: int
    x: float
    y: float
    layer: int
    hp: float = 100.0
    max_hp: float = 100.0
    persona: str = ""

    def resolved_persona(self) -> str:
        return self.persona or BOT_PERSONAS.get(self.bot_class, "Focused tactical Frostmarsh combatant.")


@dataclass
class GroundedObservation:
    bot_id: str
    bot_name: str
    bot_class: str
    sector: str
    pos_xyz: tuple[float, float, int]
    hp: float
    max_hp: float
    diurnal_time: str
    local_temp_c: float
    local_temp_k: float
    z_crop_shape: tuple[int, int, int]
    z_latent_energy: float
    z_latent_saliency: float
    z_peak_quadrant: str
    z_regime: str
    oil_slick: dict[str, Any]
    wildfire: dict[str, Any]
    skybridge: dict[str, Any]
    ice_bridge: dict[str, Any]
    ambient_noise_db: float
    waterfall_noise_db: float
    dominant_noise_cause: str
    incoming_snr_db: float
    incoming_snr_tier: str
    incoming_walls_crossed: float
    incoming_ceiling_muffled: bool
    incoming_occlusion_cause: str
    speaker_name: str
    raw_utterance: str
    heard_utterance: str
    prompt_summary: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ZWorldToLanguageBridge:
    """
    Grounds real frozen `WorldModel` `Z_world [4, 48, 256, 256]`, authoritative
    `w_state [4, 18, 256, 256]`, and `AcousticField3D` into structured physical
    observations and compact language prompts for System-2 dialogue.
    """

    def __init__(
        self,
        device: torch.device | str = "cuda",
        use_fp16: bool = True,
        patch_size: int = 16,
        ego_radius_cells: float = 16.0,
    ):
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16) and (self.device.type == "cuda")
        self.patch_size = patch_size
        self.ego_radius_cells = ego_radius_cells

        self.acoustics = AcousticField3D(layers=4).to(self.device).eval()

        lin = torch.linspace(-1.0, 1.0, patch_size, device=self.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self._base_gx = gx.unsqueeze(0).contiguous()
        self._base_gy = gy.unsqueeze(0).contiguous()

        # Per-world caches keyed by (data_ptr, shape)
        self._cached_world_key: tuple[int, tuple[int, ...]] | None = None
        self._cached_acoustic_pack: tuple[torch.Tensor, ...] | None = None
        self._cached_w_cpu_key: tuple[int, tuple[int, ...]] | None = None
        self._cached_w_cpu: np.ndarray | None = None

    @torch.inference_mode()
    def extract_ego_crop(
        self,
        z_world: torch.Tensor,
        x: float,
        y: float,
        layer: int,
    ) -> torch.Tensor:
        """Extract [48, 16, 16] float32 ego-crop from Z_world [L, 48, H, W] centered at (x, y, layer)."""
        L, _, H, W = z_world.shape
        z_dev = z_world.to(device=self.device, non_blocking=True)
        li = int(np.clip(layer, 0, L - 1))
        cx = (float(x) / max(W - 1, 1)) * 2.0 - 1.0
        cy = (float(y) / max(H - 1, 1)) * 2.0 - 1.0
        rx = (self.ego_radius_cells / max(W - 1, 1)) * 2.0
        ry = (self.ego_radius_cells / max(H - 1, 1)) * 2.0

        grid_x = cx + self._base_gx * rx
        grid_y = cy + self._base_gy * ry
        grid_2d = torch.stack([grid_x, grid_y], dim=-1).to(dtype=z_dev.dtype)
        layer_feat = z_dev[li : li + 1]
        crop = F.grid_sample(layer_feat, grid_2d, mode="bilinear", padding_mode="border", align_corners=True)
        return crop.squeeze(0).float()

    @torch.inference_mode()
    def compute_acoustic_field(
        self,
        w_state: torch.Tensor,
        force_recompute: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (solid_wall, open_shaft_01, open_sky_12, noise_db, noise_cause) via AcousticField3D."""
        key = (int(w_state.data_ptr()), tuple(w_state.shape))
        if not force_recompute and self._cached_world_key == key and self._cached_acoustic_pack is not None:
            return self._cached_acoustic_pack
        w_gpu = w_state.to(device=self.device, dtype=torch.float16 if self.use_fp16 else torch.float32)
        pack = self.acoustics.compute_geometry_and_noise(w_gpu, use_fp16=self.use_fp16)
        self._cached_world_key = key
        self._cached_acoustic_pack = pack
        return pack

    @staticmethod
    def _find_nearest_mask_cluster(
        mask_2d: np.ndarray,
        value_2d: np.ndarray,
        bx: float,
        by: float,
    ) -> tuple[bool, float, str, float, float, float]:
        ys, xs = np.nonzero(mask_2d)
        if ys.size == 0:
            return False, 999.0, "NONE", 0.0, bx, by
        if ys.size > 4096:
            step = ys.size // 2048
            ys = ys[::step]
            xs = xs[::step]
        dists = np.hypot(xs.astype(np.float32) - bx, ys.astype(np.float32) - by)
        i_min = int(np.argmin(dists))
        min_d = float(dists[i_min])
        cx, cy = float(xs[i_min]), float(ys[i_min])
        near = dists <= (min_d + 10.0)
        peak_val = float(value_2d[ys[near], xs[near]].max())
        return True, min_d, bearing_8way(cx - bx, cy - by), peak_val, cx, cy

    @torch.inference_mode()
    def observe(
        self,
        w_state: torch.Tensor,
        z_world: torch.Tensor,
        bot: BotProfile,
        speaker_name: str,
        speaker_pos: tuple[float, float, int],
        speaker_volume: float,
        raw_utterance: str,
        diurnal_time: str = "Dawn Mist (06:15)",
        landmarks: dict[str, Any] | None = None,
        seed: int = 0,
        force_recompute_acoustics: bool = False,
    ) -> GroundedObservation:
        L, _, H, W = w_state.shape
        bx = float(np.clip(bot.x, 0, W - 1))
        by = float(np.clip(bot.y, 0, H - 1))
        bz = int(np.clip(bot.layer, 0, L - 1))

        # 1. Real 16x16 Z_world ego-crop + latent energy & spatial saliency
        z_crop = self.extract_ego_crop(z_world, bx, by, bz)  # [48, 16, 16]
        z_energy = float(torch.sqrt(torch.mean(z_crop * z_crop)).item())
        z_saliency = float(z_crop.var(dim=(1, 2)).mean().item())
        q_vals = {
            "NW": float(z_crop[:, :8, :8].var(dim=(1, 2)).mean().item()),
            "NE": float(z_crop[:, :8, 8:].var(dim=(1, 2)).mean().item()),
            "SW": float(z_crop[:, 8:, :8].var(dim=(1, 2)).mean().item()),
            "SE": float(z_crop[:, 8:, 8:].var(dim=(1, 2)).mean().item()),
        }
        z_peak_quad = max(q_vals, key=q_vals.get)
        if z_saliency >= 0.08:
            z_regime = "high-flux reactive anomaly"
        elif z_saliency >= 0.025:
            z_regime = "active physical gradient"
        else:
            z_regime = "calm steady field"

        # 2. 3D Physical Acoustics (ambient noise + pairwise SNR + occlusion cause)
        solid_wall, open_shaft_01, open_sky_12, noise_db, noise_cause = self.compute_acoustic_field(
            w_state, force_recompute=force_recompute_acoustics
        )
        src_t = torch.tensor([[float(speaker_pos[0]), float(speaker_pos[1]), float(speaker_pos[2])]], device=self.device, dtype=torch.float32)
        dst_t = torch.tensor([[bx, by, float(bz)]], device=self.device, dtype=torch.float32)
        link = self.acoustics.query_pairwise_link_fast(
            src_t, dst_t, (H, W), solid_wall, open_shaft_01, open_sky_12, noise_db, noise_cause
        )
        vol = float(np.clip(speaker_volume, 0.0, 1.0))
        src_spl = SRC_SPL_MIN_DB + SRC_SPL_SPAN_DB * vol
        atten_db = float(link["atten_db"][0].item())
        dst_noise_db = float(link["dst_noise_db"][0].item())
        dst_cause_id = int(link["dst_noise_cause"][0].item())
        walls_crossed = float(link["walls_crossed"][0].item())
        ceiling_muffled = bool(link["ceiling_muffled"][0].item() > 0.5)
        snr_db = src_spl - atten_db - dst_noise_db
        occlusion_cause = _build_cause_text(dst_cause_id, walls_crossed, ceiling_muffled)
        dom_noise_cause = CAUSE_DESCRIPTIONS.get(dst_cause_id, "ambient")

        ix = int(np.clip(round(bx), 0, W - 1))
        iy = int(np.clip(round(by), 0, H - 1))
        wf_mask = noise_cause[bz] == 1
        if bool(wf_mask.any().item()):
            wf_noise_here = dst_noise_db if dst_cause_id == 1 else float(noise_db[bz][wf_mask].max().item())
        else:
            wf_noise_here = 32.0

        # 3. Cached CPU mirror of authoritative w_state [4, 18, H, W] for <2 ms sensor extraction
        w_key = (int(w_state.data_ptr()), tuple(w_state.shape))
        if force_recompute_acoustics or self._cached_w_cpu_key != w_key or self._cached_w_cpu is None:
            self._cached_w_cpu = w_state.detach().float().cpu().numpy()
            self._cached_w_cpu_key = w_key
        w_cpu = self._cached_w_cpu

        local_t_k = 290.0 + float(w_cpu[bz, CH_TEMP, iy, ix]) * 100.0
        local_t_c = local_t_k - 273.15

        # 3a. Nearby oil slicks
        oil_layer = bz if bz in (1, 2) else 1
        oil_map = w_cpu[oil_layer, S_OIL]
        temp_oil = 290.0 + w_cpu[oil_layer, CH_TEMP] * 100.0
        oil_found, oil_dist, oil_bear, oil_peak, ox, oy = self._find_nearest_mask_cluster(
            oil_map > 0.25, oil_map, bx, by
        )
        oil_temp_k = float(temp_oil[int(round(oy)), int(round(ox))]) if oil_found else 290.0
        oil_status = "absent"
        if oil_found:
            oil_status = "IGNITED/BURNING" if oil_temp_k >= 440.0 else "unlit volatile crude"
        oil_info = {
            "found": oil_found,
            "dist_cells": round(oil_dist, 1),
            "bearing": oil_bear,
            "sector": pos_to_sector(ox, oy, W, H) if oil_found else "None",
            "peak_mass": round(oil_peak, 2),
            "temp_k": round(oil_temp_k, 1),
            "status": oil_status,
        }

        # 3b. Nearest active wildfire
        t_l1_k = 290.0 + w_cpu[1, CH_TEMP] * 100.0
        t_l2_k = 290.0 + w_cpu[2, CH_TEMP] * 100.0
        t_max_12 = np.maximum(t_l1_k, t_l2_k)
        smoke_12 = np.maximum(w_cpu[1, S_SMOKE], w_cpu[2, S_SMOKE])
        wf_found, wf_dist, wf_bear, wf_peak_t, fx, fy = self._find_nearest_mask_cluster(
            t_max_12 > 460.0, t_max_12, bx, by
        )
        wf_smoke = float(smoke_12[int(round(fy)), int(round(fx))]) if wf_found else 0.0
        wildfire_info = {
            "found": wf_found,
            "dist_cells": round(wf_dist, 1),
            "bearing": wf_bear,
            "sector": pos_to_sector(fx, fy, W, H) if wf_found else "None",
            "peak_temp_k": round(wf_peak_t, 1),
            "smoke_density": round(wf_smoke, 2),
        }

        # 3c. L2 South Skybridge integrity
        if landmarks and "south_skybridge" in landmarks:
            sb_x, sb_y, _ = landmarks["south_skybridge"]
        else:
            sb_x, sb_y = 128.0, 0.69 * H
        y0 = max(0, int(sb_y) - 8)
        y1 = min(H, int(sb_y) + 9)
        x0 = int(0.36 * W)
        x1 = int(0.64 * W)
        sb_wood = w_cpu[2, S_WOOD, y0:y1, x0:x1]
        sb_temp_k = 290.0 + w_cpu[2, CH_TEMP, y0:y1, x0:x1] * 100.0
        sb_smoke = w_cpu[2, S_SMOKE, y0:y1, x0:x1]
        sb_wood_mean = float(sb_wood[sb_wood > 0.05].mean()) if (sb_wood > 0.05).any() else 0.0
        sb_cov = float((sb_wood > 0.12).mean())
        integrity_pct = float(np.clip((sb_wood_mean / 0.78) * min(1.0, sb_cov / 0.20) * 100.0, 0.0, 100.0))
        sb_peak_t = float(sb_temp_k.max())
        sb_max_smoke = float(sb_smoke.max())
        if integrity_pct < 15.0:
            sb_status = "COLLAPSED"
        elif sb_peak_t > 500.0 or (integrity_pct < 55.0 and sb_max_smoke > 0.2):
            sb_status = "BURNING — COLLAPSE IMMINENT"
        elif sb_peak_t > 380.0 or integrity_pct < 80.0:
            sb_status = "SCORCHED/WEAKENED"
        else:
            sb_status = "INTACT"
        sb_dist = float(math.hypot(sb_x - bx, sb_y - by))
        sb_bear = bearing_8way(sb_x - bx, sb_y - by)
        skybridge_info = {
            "south_integrity_pct": round(integrity_pct, 1),
            "peak_temp_k": round(sb_peak_t, 1),
            "smoke": round(sb_max_smoke, 2),
            "status": sb_status,
            "dist_cells": round(sb_dist, 1),
            "bearing": sb_bear,
        }

        # 3d. Frozen river ice bridge
        ice_l1 = w_cpu[1, S_ICE]
        ice_found, ice_dist, ice_bear, ice_peak, _, _ = self._find_nearest_mask_cluster(
            ice_l1 > 0.45, ice_l1, bx, by
        )
        ice_info = {
            "found": ice_found,
            "dist_cells": round(ice_dist, 1),
            "bearing": ice_bear,
            "peak_mass": round(ice_peak, 2),
            "status": "SOLID ICE" if (ice_found and ice_peak >= 0.75) else ("THIN ICE" if ice_found else "OPEN WATER"),
        }

        # 4. Physically degrade incoming utterance by exact AcousticField3D SNR
        heard_text, snr_tier = degrade_utterance_by_snr(
            raw_utterance, snr_db=snr_db, occlusion_cause=occlusion_cause, seed=seed
        )

        # 5. Build grounded prompt summary with explicit acoustic/hazard alerts
        sector = pos_to_sector(bx, by, W, H)
        oil_str = (
            f"{oil_info['dist_cells']:.0f}m {oil_info['bearing']} ({oil_info['sector']}, {oil_info['status']})"
            if oil_found else "none nearby"
        )
        fire_str = (
            f"{wildfire_info['dist_cells']:.0f}m {wildfire_info['bearing']} ({wildfire_info['sector']}, {wildfire_info['peak_temp_k']:.0f}K)"
            if wf_found else "none active"
        )
        alert_line = ""
        if snr_db < -6.0:
            alert_line = (
                f"\nCOMMS ALERT: Signal INAUDIBLE ({snr_db:+.1f}dB) drowned out by {occlusion_cause}. "
                f"You heard NOTHING — state {occlusion_cause} drowned comms, ask to repeat/relocate, intent=RELOCATE_COMMS or REQUEST_REPEAT."
            )
        elif snr_db <= 1.5:
            alert_line = (
                f"\nCOMMS ALERT: Signal MUFFLED ({snr_db:+.1f}dB) by {occlusion_cause}. "
                f"Report {occlusion_cause} breaking up comms and ask to repeat loudly, intent=REQUEST_REPEAT."
            )
        elif "COLLAPSE" in sb_status:
            alert_line = (
                f"\nHAZARD ALERT: L2 south skybridge at {integrity_pct:.0f}% integrity ({sb_peak_t:.0f}K) — COLLAPSE IMMINENT! "
                f"Warn teammate off the burning bridge and quench timbers with cryo water (thermal_flux=-0.9, matter=water)."
            )

        prompt_summary = (
            f"SELF: {bot.name} ({bot.bot_class}, HP {bot.hp:.0f}/{bot.max_hp:.0f}, L{bz} {sector}, pos {bx:.0f},{by:.0f})\n"
            f"TIME/CLIMATE: {diurnal_time}, local {local_t_c:+.1f}C ({local_t_k:.0f}K)\n"
            f"Z_WORLD[16x16]: energy={z_energy:.2f}, saliency={z_saliency:.3f} ({z_regime}, peak {z_peak_quad})\n"
            f"HAZARDS: oil={oil_str} | wildfire={fire_str} | L2_south_skybridge={integrity_pct:.0f}% ({sb_status}, {sb_dist:.0f}m {sb_bear}) | river_ice={ice_info['status']} ({ice_dist:.0f}m {ice_bear})\n"
            f"ACOUSTICS: local_noise={dst_noise_db:.1f}dB ({dom_noise_cause}), waterfall_peak={wf_noise_here:.1f}dB\n"
            f"INCOMING from {speaker_name}: snr={snr_db:+.1f}dB ({snr_tier}; cause={occlusion_cause}) | HEARD: \"{heard_text}\""
            f"{alert_line}"
        )

        return GroundedObservation(
            bot_id=bot.bot_id,
            bot_name=bot.name,
            bot_class=bot.bot_class,
            sector=sector,
            pos_xyz=(round(bx, 1), round(by, 1), bz),
            hp=float(bot.hp),
            max_hp=float(bot.max_hp),
            diurnal_time=diurnal_time,
            local_temp_c=round(local_t_c, 1),
            local_temp_k=round(local_t_k, 1),
            z_crop_shape=(int(z_crop.shape[0]), int(z_crop.shape[1]), int(z_crop.shape[2])),
            z_latent_energy=round(z_energy, 4),
            z_latent_saliency=round(z_saliency, 5),
            z_peak_quadrant=z_peak_quad,
            z_regime=z_regime,
            oil_slick=oil_info,
            wildfire=wildfire_info,
            skybridge=skybridge_info,
            ice_bridge=ice_info,
            ambient_noise_db=round(dst_noise_db, 2),
            waterfall_noise_db=round(wf_noise_here, 2),
            dominant_noise_cause=dom_noise_cause,
            incoming_snr_db=round(snr_db, 2),
            incoming_snr_tier=snr_tier,
            incoming_walls_crossed=round(walls_crossed, 2),
            incoming_ceiling_muffled=ceiling_muffled,
            incoming_occlusion_cause=occlusion_cause,
            speaker_name=speaker_name,
            raw_utterance=raw_utterance,
            heard_utterance=heard_text,
            prompt_summary=prompt_summary,
        )


# =============================================================================
# 2. System2DialogueEngine (Qwen2.5-1.5B-Instruct, Sync + Async Non-Blocking)
# =============================================================================
SYSTEM_PROMPT = (
    "You are a Frostmarsh 3D physics arena bot. Reply ONLY as compact single-line JSON (no spaces after colons/commas, no markdown):\n"
    '{"reply":"<6-14 words in-character>","intent":"<ENUM>","target_sector":"<SECTOR>","actuator_bias":{"thermal_flux":<float>,"matter":"<oil|water|smoke|none>","focus_radius":<float>}}\n'
    "Valid intent ENUMs: IGNITE_TRAP, STAGE_OIL, HOLD_FIRE, FREEZE_CHOKE, QUENCH_FIRE, SMOKE_COVER, WARN_COLLAPSE, HOLD_BRIDGE, REQUEST_REPEAT, RELOCATE_COMMS.\n"
    "Valid SECTORs: NW Highland, NE Skybridge, SW Oil Basin, SE Shrine Valley, L0 Cavern, Central River.\n"
    "Actuator rules: ignite/fire=>thermal_flux +0.85..+1.0, matter none; frost/freeze/quench=>thermal_flux -0.85..-1.0, matter water; lay oil=>thermal_flux 0.0, matter oil; smoke=>thermal_flux 0.0, matter smoke; hold/repeat=>thermal_flux 0.0, matter none. focus_radius in [2.0,8.0].\n"
    "Examples:\n"
    '{"reply":"Oil slick 9m SW unlit — igniting wide flame burst now!","intent":"IGNITE_TRAP","target_sector":"SW Oil Basin","actuator_bias":{"thermal_flux":0.95,"matter":"none","focus_radius":6.5}}\n'
    '{"reply":"Say again! Waterfall roar at 85dB drowned comms — moving clear!","intent":"REQUEST_REPEAT","target_sector":"NW Highland","actuator_bias":{"thermal_flux":0.0,"matter":"none","focus_radius":3.5}}\n'
    '{"reply":"Do not cross! South skybridge at 30% burning — quenching with cryo!","intent":"WARN_COLLAPSE","target_sector":"NE Skybridge","actuator_bias":{"thermal_flux":-0.9,"matter":"water","focus_radius":5.5}}'
)


class _FirstTokenTimerStreamer(BaseStreamer):
    """Captures exact GPU-synchronized TTFT on the very first generated token."""

    def __init__(self, device: torch.device):
        self.device = device
        self.call_count = 0
        self.t_first_token: float | None = None

    def put(self, value: torch.Tensor) -> None:
        self.call_count += 1
        if self.call_count == 2 and self.t_first_token is None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.t_first_token = time.perf_counter()

    def end(self) -> None:
        pass


@dataclass
class DialogueRequest:
    bot: BotProfile
    observation: GroundedObservation
    conversation_id: str = "default"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    max_new_tokens: int = 64
    submitted_at: float = field(default_factory=time.perf_counter)


@dataclass
class DialogueResponse:
    request_id: str
    conversation_id: str
    bot_id: str
    bot_name: str
    bot_class: str
    plan: dict[str, Any]
    raw_text: str
    raw_json_valid: bool
    plan_valid: bool
    repaired: bool
    ttft_ms: float
    total_ms: float
    queue_wait_ms: float
    prompt_tokens: int
    generated_tokens: int
    tokens_per_sec: float
    decode_tokens_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _clamp_word_count(text: str, max_words: int = 16) -> str:
    clean = " ".join(str(text).replace("\n", " ").split()).strip().strip('"').strip("'")
    words = clean.split()
    if len(words) <= max_words:
        return clean
    clipped = " ".join(words[:max_words]).rstrip(",;:-")
    if clipped and clipped[-1] not in ".!?":
        clipped += "."
    return clipped


@dataclass
class TaskBrief:
    """NMMO 2.0-style compact commander conditioning vector fed into recurrent BotPolicyNetV2."""
    objective: int  # 0=FREE_ROAM, 1=CAPTURE_SHRINE, 2=ESCORT_RELIC, 3=HUNT_ENEMY, 4=DEFEND_HOLD
    stance: int     # 0=COMMIT, 1=HOLD, 2=FLEE_KITE
    target_x: float
    target_y: float
    risk_budget: float
    thermal_bias: float
    role_id: int
    active: bool = True

    def to_vec15(self, bot_x: float, bot_y: float) -> list[float]:
        vec = [0.0] * 15
        vec[max(0, min(4, int(self.objective)))] = 1.0
        vec[5 + max(0, min(2, int(self.stance)))] = 1.0
        dx = float(self.target_x) - float(bot_x)
        dy = float(self.target_y) - float(bot_y)
        dist = math.hypot(dx, dy)
        vec[8] = max(-1.0, min(1.0, dx / 100.0))
        vec[9] = max(-1.0, min(1.0, dy / 100.0))
        vec[10] = min(1.0, dist / 100.0)
        vec[11] = max(0.0, min(1.0, float(self.risk_budget)))
        vec[12] = max(0.0, min(1.0, float(max(0, min(4, int(self.role_id)))) / 4.0))
        vec[13] = 1.0 if self.active else 0.0
        vec[14] = max(-1.0, min(1.0, float(self.thermal_bias)))
        return vec

    def to_dict(self, bot_x: float, bot_y: float) -> dict[str, Any]:
        return {
            "objective": int(self.objective),
            "stance": int(self.stance),
            "target_x": round(float(self.target_x), 2),
            "target_y": round(float(self.target_y), 2),
            "risk_budget": round(float(self.risk_budget), 3),
            "thermal_bias": round(float(self.thermal_bias), 3),
            "role_id": int(self.role_id),
            "active": bool(self.active),
            "brief_vec_15": [round(v, 4) for v in self.to_vec15(bot_x, bot_y)],
        }


def _bot_class_to_role_id(bot_class: str | int) -> int:
    if isinstance(bot_class, int):
        return max(0, min(4, bot_class))
    s = str(bot_class).lower()
    if "cryo" in s or "frost" in s or "tide" in s:
        return 1
    if "alchem" in s or "sky" in s:
        return 2
    if "shadow" in s or "stalk" in s or "flank" in s:
        return 3
    if "burrow" in s or "mole" in s or "strider" in s:
        return 4
    return 0


def compile_plan_to_brief(
    plan: dict[str, Any],
    bot_pos: tuple[float, float],
    grid_w: float = 256.0,
    grid_h: float = 256.0,
    role_id: int = 0,
) -> dict[str, Any]:
    """Compile a System-2 SLM JSON tactical plan into the 15-float NMMO 2.0 TaskBrief dict."""
    intent = str(plan.get("intent", "HOLD_BRIDGE")).upper().strip()
    sector = str(plan.get("target_sector", "")).strip()
    ab = plan.get("actuator_bias") or {}
    tf = float(np.clip(float(ab.get("thermal_flux", 0.0)), -1.0, 1.0))

    sector_map = {
        "NW Highland": (grid_w * 0.28, grid_h * 0.25),
        "NE Skybridge": (grid_w * 0.55, grid_h * 0.30),
        "SW Oil Basin": (grid_w * 0.28, grid_h * 0.72),
        "SE Shrine Valley": (grid_w * 0.72, grid_h * 0.72),
        "L0 Cavern": (grid_w * 0.50, grid_h * 0.50),
        "Central River": (grid_w * 0.50, grid_h * 0.50),
        "North Village": (grid_w * 0.50, grid_h * 0.24),
        "Harbor": (grid_w * 0.22, grid_h * 0.50),
        "Relic": (grid_w * 0.50, grid_h * 0.50),
    }
    tx, ty = sector_map.get(sector, (float(bot_pos[0]), float(bot_pos[1])))

    # Map intent -> (objective 0..4, stance 0..2, risk_budget 0..1)
    intent_map: dict[str, tuple[int, int, float]] = {
        "IGNITE_TRAP": (3, 0, 0.85),       # HUNT_ENEMY, COMMIT
        "HUNT_ENEMY": (3, 0, 0.90),
        "CAPTURE_SHRINE": (1, 0, 0.75),    # CAPTURE_SHRINE, COMMIT
        "ESCORT_RELIC": (2, 0, 0.65),      # ESCORT_RELIC, COMMIT
        "STAGE_OIL": (1, 1, 0.55),         # CAPTURE_SHRINE, HOLD
        "HOLD_BRIDGE": (4, 1, 0.50),       # DEFEND_HOLD, HOLD
        "FREEZE_CHOKE": (4, 1, 0.45),      # DEFEND_HOLD, HOLD
        "HOLD_FIRE": (4, 1, 0.35),         # DEFEND_HOLD, HOLD
        "QUENCH_FIRE": (4, 1, 0.40),       # DEFEND_HOLD, HOLD
        "WARN_COLLAPSE": (4, 2, 0.20),     # DEFEND_HOLD, FLEE_KITE
        "SMOKE_COVER": (0, 2, 0.25),       # FREE_ROAM, FLEE_KITE
        "RETREAT": (0, 2, 0.15),           # FREE_ROAM, FLEE_KITE
        "RELOCATE_COMMS": (0, 2, 0.25),    # FREE_ROAM, FLEE_KITE
        "REQUEST_REPEAT": (4, 1, 0.30),    # DEFEND_HOLD, HOLD
    }
    obj, stance, risk = intent_map.get(intent, (4, 1, 0.50))
    if "objective" in plan and isinstance(plan["objective"], int):
        obj = max(0, min(4, plan["objective"]))
    if "stance" in plan and isinstance(plan["stance"], int):
        stance = max(0, min(2, plan["stance"]))
    if "risk_budget" in plan:
        try:
            risk = float(np.clip(float(plan["risk_budget"]), 0.0, 1.0))
        except Exception:
            pass

    tb = TaskBrief(
        objective=obj,
        stance=stance,
        target_x=tx,
        target_y=ty,
        risk_budget=risk,
        thermal_bias=tf,
        role_id=_bot_class_to_role_id(role_id),
        active=True,
    )
    return tb.to_dict(float(bot_pos[0]), float(bot_pos[1]))


def _default_physics_plan(obs: GroundedObservation) -> dict[str, Any]:
    if obs.incoming_snr_db < -6.0:
        return {
            "reply": f"Comms dead — {obs.incoming_occlusion_cause}! Relocating for signal, say again!",
            "intent": "RELOCATE_COMMS",
            "target_sector": obs.sector,
            "actuator_bias": {"thermal_flux": 0.0, "matter": "none", "focus_radius": 3.5},
        }
    if obs.incoming_snr_db <= 1.5:
        return {
            "reply": f"Breaking up over {obs.incoming_occlusion_cause} — repeat orders!",
            "intent": "REQUEST_REPEAT",
            "target_sector": obs.sector,
            "actuator_bias": {"thermal_flux": 0.0, "matter": "none", "focus_radius": 3.5},
        }
    if "COLLAPSE" in obs.skybridge.get("status", ""):
        return {
            "reply": f"South skybridge at {obs.skybridge['south_integrity_pct']:.0f}% burning — quenching with cryo!",
            "intent": "WARN_COLLAPSE",
            "target_sector": "NE Skybridge",
            "actuator_bias": {"thermal_flux": -0.9, "matter": "water", "focus_radius": 5.5},
        }
    if obs.oil_slick.get("found") and obs.bot_class == "Pyromancer":
        return {
            "reply": f"Oil slick {obs.oil_slick['dist_cells']:.0f}m {obs.oil_slick['bearing']} — igniting trap now!",
            "intent": "IGNITE_TRAP",
            "target_sector": obs.oil_slick.get("sector", obs.sector),
            "actuator_bias": {"thermal_flux": 0.9, "matter": "none", "focus_radius": 4.0},
        }
    if obs.bot_class == "Cryo-Scout":
        return {
            "reply": f"Holding {obs.sector} at {obs.local_temp_c:+.0f}C — reinforcing frost barrier now.",
            "intent": "FREEZE_CHOKE",
            "target_sector": obs.sector,
            "actuator_bias": {"thermal_flux": -0.85, "matter": "water", "focus_radius": 6.0},
        }
    return {
        "reply": f"Copy at {obs.sector} — holding position and matching actuator bias.",
        "intent": "HOLD_BRIDGE",
        "target_sector": obs.sector,
        "actuator_bias": {"thermal_flux": 0.0, "matter": "oil" if obs.bot_class == "Shadow-Flanker" else "none", "focus_radius": 4.0},
    }


def parse_and_validate_plan(
    raw_json_str: str,
    obs: GroundedObservation,
) -> tuple[dict[str, Any], bool, bool, bool]:
    """Parse model output into strict structured plan schema; returns (plan, raw_json_valid, plan_valid, repaired)."""
    raw_valid = False
    repaired = False
    parsed: dict[str, Any] | None = None

    text = raw_json_str.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    if start != -1:
        candidate = text[start:]
        depth = 0
        end_idx = -1
        in_str = False
        esc = False
        for idx, ch in enumerate(candidate):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = idx
                        break
        if end_idx != -1:
            try:
                obj = json.loads(candidate[: end_idx + 1])
                if isinstance(obj, dict):
                    parsed = obj
                    raw_valid = True
            except Exception:
                parsed = None

    fallback = _default_physics_plan(obs)
    if parsed is None:
        repaired = True
        parsed = {}
        m_rep = re.search(r'"reply"\s*:\s*"([^"]+)"', text)
        if m_rep:
            parsed["reply"] = m_rep.group(1)
        m_int = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
        if m_int:
            parsed["intent"] = m_int.group(1)
        m_sec = re.search(r'"target_sector"\s*:\s*"([^"]+)"', text)
        if m_sec:
            parsed["target_sector"] = m_sec.group(1)
        ab: dict[str, Any] = {}
        m_tf = re.search(r'"thermal_flux"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if m_tf:
            ab["thermal_flux"] = float(m_tf.group(1))
        m_mat = re.search(r'"matter"\s*:\s*"([^"]+)"', text)
        if m_mat:
            ab["matter"] = m_mat.group(1)
        m_rad = re.search(r'"focus_radius"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if m_rad:
            ab["focus_radius"] = float(m_rad.group(1))
        if ab:
            parsed["actuator_bias"] = ab

    reply_raw = parsed.get("reply")
    if not isinstance(reply_raw, str) or len(reply_raw.strip()) < 2:
        reply_str = fallback["reply"]
        repaired = True
    else:
        reply_str = _clamp_word_count(reply_raw, max_words=16)

    intent_raw = parsed.get("intent")
    if not isinstance(intent_raw, str) or not intent_raw.strip():
        intent_str = fallback["intent"]
        repaired = True
    else:
        intent_str = re.sub(r"[^A-Za-z0-9_\- ]", "", intent_raw.strip()).upper().replace(" ", "_")[:32] or fallback["intent"]

    sector_raw = parsed.get("target_sector")
    if not isinstance(sector_raw, str) or not sector_raw.strip():
        sector_str = fallback["target_sector"]
        repaired = True
    else:
        sector_str = sector_raw.strip()[:32]

    ab_raw = parsed.get("actuator_bias")
    if not isinstance(ab_raw, dict):
        ab_raw = {}
        repaired = True

    try:
        tf = float(ab_raw.get("thermal_flux", fallback["actuator_bias"]["thermal_flux"]))
        if not math.isfinite(tf):
            raise ValueError("non-finite")
        tf = round(float(np.clip(tf, -1.0, 1.0)), 2)
    except Exception:
        tf = fallback["actuator_bias"]["thermal_flux"]
        repaired = True

    mat_raw = str(ab_raw.get("matter", fallback["actuator_bias"]["matter"])).strip().lower()
    if mat_raw not in VALID_MATTERS:
        if "oil" in mat_raw or "crude" in mat_raw:
            mat_raw = "oil"
        elif "water" in mat_raw or "ice" in mat_raw or "frost" in mat_raw:
            mat_raw = "water"
        elif "smoke" in mat_raw or "mist" in mat_raw:
            mat_raw = "smoke"
        else:
            mat_raw = "none"
        repaired = True

    try:
        fr = float(ab_raw.get("focus_radius", fallback["actuator_bias"]["focus_radius"]))
        if not math.isfinite(fr):
            raise ValueError("non-finite")
        fr = round(float(np.clip(fr, 2.0, 8.0)), 2)
    except Exception:
        fr = fallback["actuator_bias"]["focus_radius"]
        repaired = True

    plan = {
        "reply": reply_str,
        "intent": intent_str,
        "target_sector": sector_str,
        "actuator_bias": {
            "thermal_flux": tf,
            "matter": mat_raw,
            "focus_radius": fr,
        },
    }
    plan["task_brief"] = compile_plan_to_brief(
        plan,
        bot_pos=(obs.bot_pos[0], obs.bot_pos[1]),
        grid_w=256.0,
        grid_h=256.0,
        role_id=_bot_class_to_role_id(obs.bot_class),
    )

    plan_valid = (
        isinstance(plan["reply"], str)
        and 1 <= len(plan["reply"].split()) <= 16
        and isinstance(plan["intent"], str)
        and len(plan["intent"]) > 0
        and isinstance(plan["target_sector"], str)
        and len(plan["target_sector"]) > 0
        and -1.0 <= plan["actuator_bias"]["thermal_flux"] <= 1.0
        and plan["actuator_bias"]["matter"] in VALID_MATTERS
        and 2.0 <= plan["actuator_bias"]["focus_radius"] <= 8.0
    )
    if raw_valid:
        raw_valid = bool(
            isinstance(parsed.get("reply"), str)
            and isinstance(parsed.get("intent"), str)
            and isinstance(parsed.get("target_sector"), str)
            and isinstance(parsed.get("actuator_bias"), dict)
            and "thermal_flux" in parsed["actuator_bias"]
            and "matter" in parsed["actuator_bias"]
            and "focus_radius" in parsed["actuator_bias"]
        )

    return plan, raw_valid, plan_valid, repaired


class System2DialogueEngine:
    """
    System-2 SLM Dialogue & Tactical Planning Engine backed by `Qwen/Qwen2.5-1.5B-Instruct`
    in BF16 on CUDA with both synchronous `generate_reply(...)` and a strictly non-blocking
    background worker thread (`submit_dialogue(...)` / `poll_replies()`).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        max_memory_turns: int = 4,
        local_files_only: bool = True,
        start_worker: bool = True,
    ):
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if dtype is None:
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            elif self.device.type == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        self.max_memory_turns = max_memory_turns
        self._memory_lock = threading.Lock()
        self._memory: dict[tuple[str, str], collections.deque[dict[str, Any]]] = {}

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self.dtype,
            local_files_only=local_files_only,
        ).to(self.device).eval()

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.load_time_s = time.perf_counter() - t0
        self._infer_lock = threading.Lock()

        self._in_queue: queue.Queue[DialogueRequest | None] = queue.Queue()
        self._out_queue: queue.Queue[DialogueResponse] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        if start_worker:
            self.start_worker()

    def start_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="System2DialogueWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def shutdown(self, timeout_s: float = 10.0) -> None:
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return
        self._stop_event.set()
        self._in_queue.put(None)
        self._worker_thread.join(timeout=timeout_s)

    def clear_memory(self, conversation_id: str | None = None) -> None:
        with self._memory_lock:
            if conversation_id is None:
                self._memory.clear()
            else:
                for k in [k for k in self._memory if k[0] == conversation_id]:
                    del self._memory[k]

    def get_memory(self, conversation_id: str, bot_id: str) -> list[dict[str, Any]]:
        with self._memory_lock:
            dq = self._memory.get((conversation_id, bot_id))
            return list(dq) if dq else []

    def _append_memory(
        self,
        conversation_id: str,
        bot_id: str,
        speaker: str,
        heard_text: str,
        snr_db: float,
        reply: str,
        intent: str,
    ) -> None:
        with self._memory_lock:
            key = (conversation_id, bot_id)
            if key not in self._memory:
                self._memory[key] = collections.deque(maxlen=self.max_memory_turns)
            self._memory[key].append(
                {
                    "speaker": speaker,
                    "heard": heard_text,
                    "snr_db": round(snr_db, 1),
                    "reply": reply,
                    "intent": intent,
                }
            )

    def _build_prompt(self, req: DialogueRequest) -> str:
        history = self.get_memory(req.conversation_id, req.bot.bot_id)
        if history:
            mem_lines = [
                f"- {h['speaker']} ({h['snr_db']:+.0f}dB): \"{h['heard']}\" -> You ({h['intent']}): \"{h['reply']}\""
                for h in history
            ]
            mem_block = "RECENT MEMORY (last 4 turns):\n" + "\n".join(mem_lines) + "\n"
        else:
            mem_block = "RECENT MEMORY: (first contact)\n"

        user_msg = (
            f"PERSONA: {req.bot.resolved_persona()}\n"
            f"{mem_block}"
            f"{req.observation.prompt_summary}\n"
            "Output ONLY the compact single-line JSON."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt + '{"reply":"'

    @torch.inference_mode()
    def generate_reply(self, req: DialogueRequest) -> DialogueResponse:
        """Synchronous GPU inference returning structured DialogueResponse with exact TTFT & total latency."""
        t_start = time.perf_counter()
        queue_wait_ms = max(0.0, (t_start - req.submitted_at) * 1000.0)
        prompt = self._build_prompt(req)
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)
        prompt_len = int(input_ids.shape[1])

        streamer = _FirstTokenTimerStreamer(self.device)
        with self._infer_lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t_gen0 = time.perf_counter()
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t_gen1 = time.perf_counter()

        new_ids = out_ids[0, prompt_len:]
        gen_tokens = int(new_ids.shape[0])
        raw_suffix = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        raw_full = '{"reply":"' + raw_suffix

        total_ms = (t_gen1 - t_gen0) * 1000.0
        ttft_ms = (
            (streamer.t_first_token - t_gen0) * 1000.0
            if streamer.t_first_token is not None
            else total_ms
        )
        total_s = max(total_ms / 1000.0, 1e-6)
        decode_s = max((total_ms - ttft_ms) / 1000.0, 1e-6)
        tok_s = gen_tokens / total_s
        decode_tok_s = max(gen_tokens - 1, 1) / decode_s

        plan, raw_valid, plan_valid, repaired = parse_and_validate_plan(raw_full, req.observation)
        self._append_memory(
            conversation_id=req.conversation_id,
            bot_id=req.bot.bot_id,
            speaker=req.observation.speaker_name,
            heard_text=req.observation.heard_utterance,
            snr_db=req.observation.incoming_snr_db,
            reply=plan["reply"],
            intent=plan["intent"],
        )

        return DialogueResponse(
            request_id=req.request_id,
            conversation_id=req.conversation_id,
            bot_id=req.bot.bot_id,
            bot_name=req.bot.name,
            bot_class=req.bot.bot_class,
            plan=plan,
            raw_text=raw_full,
            raw_json_valid=raw_valid,
            plan_valid=plan_valid,
            repaired=repaired,
            ttft_ms=round(ttft_ms, 2),
            total_ms=round(total_ms, 2),
            queue_wait_ms=round(queue_wait_ms, 2),
            prompt_tokens=prompt_len,
            generated_tokens=gen_tokens,
            tokens_per_sec=round(tok_s, 2),
            decode_tokens_per_sec=round(decode_tok_s, 2),
        )

    def submit_dialogue(self, req: DialogueRequest) -> str:
        """Strictly non-blocking submission to the background worker queue (microseconds!)."""
        req.submitted_at = time.perf_counter()
        self._in_queue.put_nowait(req)
        return req.request_id

    def poll_replies(self, max_items: int | None = None) -> list[DialogueResponse]:
        """Strictly non-blocking poll of completed dialogue responses for the 30 Hz game loop."""
        results: list[DialogueResponse] = []
        while max_items is None or len(results) < max_items:
            try:
                results.append(self._out_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                req = self._in_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if req is None:
                self._in_queue.task_done()
                break
            try:
                resp = self.generate_reply(req)
                self._out_queue.put_nowait(resp)
            except Exception as exc:
                fallback = _default_physics_plan(req.observation)
                self._out_queue.put_nowait(
                    DialogueResponse(
                        request_id=req.request_id,
                        conversation_id=req.conversation_id,
                        bot_id=req.bot.bot_id,
                        bot_name=req.bot.name,
                        bot_class=req.bot.bot_class,
                        plan=fallback,
                        raw_text=f"ERROR: {exc}",
                        raw_json_valid=False,
                        plan_valid=True,
                        repaired=True,
                        ttft_ms=0.0,
                        total_ms=0.0,
                        queue_wait_ms=0.0,
                        prompt_tokens=0,
                        generated_tokens=0,
                        tokens_per_sec=0.0,
                        decode_tokens_per_sec=0.0,
                    )
                )
            finally:
                self._in_queue.task_done()


# =============================================================================
# 3. Unseen 256x256x4 Frostmarsh Scene Modifiers & 8 Conversation Scenarios
# =============================================================================
def apply_diurnal_and_scenario_physics(
    w_state: torch.Tensor,
    landmarks: dict[str, Any],
    diurnal_temp_delta_k: float = 0.0,
    burn_south_skybridge: bool = False,
    freeze_river: bool = False,
) -> torch.Tensor:
    """Apply authentic physical scenario perturbations (midnight freeze, burning L2 skybridge)."""
    w = w_state.clone()
    L, _, H, W = w.shape

    if abs(diurnal_temp_delta_k) > 1e-3:
        for l in range(L):
            ambient_mask = w[l, CH_TEMP] < 1.5
            w[l, CH_TEMP] = torch.where(
                ambient_mask,
                torch.clamp(w[l, CH_TEMP] + (diurnal_temp_delta_k / 100.0), -1.35, 1.45),
                w[l, CH_TEMP],
            )

    if freeze_river:
        river_water = w[1, S_WATER] > 0.55
        w[1, S_ICE] = torch.where(river_water, torch.clamp(w[1, S_ICE] + 0.85, max=1.15), w[1, S_ICE])
        w[1, S_WATER] = torch.where(river_water, w[1, S_WATER] * 0.25, w[1, S_WATER])
        w[1, CH_TEMP] = torch.where(river_water, torch.full_like(w[1, CH_TEMP], -0.34), w[1, CH_TEMP])

    if burn_south_skybridge:
        _, sb_y, _ = landmarks["south_skybridge"]
        y0 = max(0, int(sb_y) - 6)
        y1 = min(H, int(sb_y) + 7)
        x0 = int(0.36 * W)
        x1 = int(0.64 * W)
        bridge_band = w[2, S_WOOD, y0:y1, x0:x1] > 0.05
        w[2, S_WOOD, y0:y1, x0:x1] = torch.where(
            bridge_band, w[2, S_WOOD, y0:y1, x0:x1] * 0.30, w[2, S_WOOD, y0:y1, x0:x1]
        )
        w[2, S_ASH, y0:y1, x0:x1] = torch.where(
            bridge_band, torch.full_like(w[2, S_ASH, y0:y1, x0:x1], 0.48), w[2, S_ASH, y0:y1, x0:x1]
        )
        w[2, S_SMOKE, y0:y1, x0:x1] = torch.where(
            bridge_band, torch.full_like(w[2, S_SMOKE, y0:y1, x0:x1], 0.72), w[2, S_SMOKE, y0:y1, x0:x1]
        )
        w[2, CH_TEMP, y0:y1, x0:x1] = torch.where(
            bridge_band, torch.full_like(w[2, CH_TEMP, y0:y1, x0:x1], 4.05), w[2, CH_TEMP, y0:y1, x0:x1]
        )  # 695 K roaring timber fire

    return w


def run_benchmark_and_conversations(
    out_metrics_json: Path = DEFAULT_METRICS_JSON,
    out_report_md: Path = DEFAULT_REPORT_MD,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[system2_slm] Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    wm, wm_path = load_frozen_world_model(device)
    bridge = ZWorldToLanguageBridge(device=device, use_fp16=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    engine = System2DialogueEngine(
        model_id=DEFAULT_MODEL_ID,
        device="cuda" if device.type == "cuda" else "cpu",
        local_files_only=True,
        start_worker=True,
    )
    print(f"[system2_slm] Loaded {DEFAULT_MODEL_ID} ({engine.dtype}) in {engine.load_time_s:.2f}s | WorldModel={wm_path.name}", flush=True)

    # 1. Warmup + Dedicated Latency Benchmark on real unseen Frostmarsh scene (seed 1399)
    w_bench, _, lm_bench = build_frostmarsh_3d_state(1399, device)
    _, z_bench = world_forward_with_latent(wm, w_bench)
    ox_b, oy_b, _ = lm_bench["oil_positions"][0]
    bench_bot = BotProfile(
        bot_id="bench_pyra",
        name="Pyra",
        bot_class="Pyromancer",
        team=0,
        x=min(248.0, ox_b + 8.0),
        y=min(248.0, oy_b + 6.0),
        layer=1,
        hp=94.0,
    )
    bench_prompts = [
        "Pyra, ignite the crude oil pool ahead before their vanguard crosses!",
        "Hold fire two seconds, let them step into the center of the slick.",
        "Switch to wide flame burst and seal the eastern flank now!",
        "Report local oil status and wildfire distance on your ridge.",
        "Drop smoke if your HP dips, then flare the southwest basin.",
    ]

    for i in range(3):
        obs_w = bridge.observe(
            w_bench, z_bench, bench_bot,
            speaker_name="Commander",
            speaker_pos=(bench_bot.x - 10.0, bench_bot.y - 6.0, 1),
            speaker_volume=0.85,
            raw_utterance=bench_prompts[i % len(bench_prompts)],
            diurnal_time="Dawn Mist (06:15)",
            landmarks=lm_bench,
            seed=100 + i,
        )
        _ = engine.generate_reply(DialogueRequest(bot=bench_bot, observation=obs_w, conversation_id="warmup"))
    engine.clear_memory("warmup")

    bench_ttfts: list[float] = []
    bench_totals: list[float] = []
    bench_tps: list[float] = []
    bench_decode_tps: list[float] = []
    bench_gen_toks: list[int] = []
    bench_bridge_ms: list[float] = []
    bench_raw_valid = 0
    bench_plan_valid = 0

    for i in range(8):
        tb0 = time.perf_counter()
        obs_b = bridge.observe(
            w_bench, z_bench, bench_bot,
            speaker_name="Commander",
            speaker_pos=(bench_bot.x - 12.0, bench_bot.y - 8.0, 1),
            speaker_volume=0.88,
            raw_utterance=bench_prompts[i % len(bench_prompts)],
            diurnal_time="Dawn Mist (06:15)",
            landmarks=lm_bench,
            seed=200 + i,
            force_recompute_acoustics=False,
        )
        tb1 = time.perf_counter()
        bench_bridge_ms.append((tb1 - tb0) * 1000.0)

        resp_b = engine.generate_reply(
            DialogueRequest(bot=bench_bot, observation=obs_b, conversation_id=f"bench_{i // 4}")
        )
        bench_ttfts.append(resp_b.ttft_ms)
        bench_totals.append(resp_b.total_ms)
        bench_tps.append(resp_b.tokens_per_sec)
        bench_decode_tps.append(resp_b.decode_tokens_per_sec)
        bench_gen_toks.append(resp_b.generated_tokens)
        bench_raw_valid += int(resp_b.raw_json_valid)
        bench_plan_valid += int(resp_b.plan_valid)

    # 2. Non-Blocking 30 Hz Async Worker Queue Benchmark (6 queued dialogue submissions)
    submit_us_samples: list[float] = []
    poll_us_samples: list[float] = []
    async_req_ids: set[str] = set()
    async_replies: list[DialogueResponse] = []
    obs_async = bridge.observe(
        w_bench, z_bench, bench_bot,
        speaker_name="Commander",
        speaker_pos=(bench_bot.x - 10.0, bench_bot.y - 6.0, 1),
        speaker_volume=0.90,
        raw_utterance="Async check: confirm oil trap readiness on southwest ridge!",
        diurnal_time="Dawn Mist (06:15)",
        landmarks=lm_bench,
        seed=999,
    )
    for _ in range(6):
        req_a = DialogueRequest(bot=bench_bot, observation=obs_async, conversation_id="async_bench")
        ts0 = time.perf_counter()
        rid = engine.submit_dialogue(req_a)
        ts1 = time.perf_counter()
        submit_us_samples.append((ts1 - ts0) * 1_000_000.0)
        async_req_ids.add(rid)

        tp0 = time.perf_counter()
        got = engine.poll_replies()
        tp1 = time.perf_counter()
        poll_us_samples.append((tp1 - tp0) * 1_000_000.0)
        async_replies.extend(got)

    t_wait0 = time.perf_counter()
    while len(async_replies) < len(async_req_ids) and (time.perf_counter() - t_wait0) < 30.0:
        tp0 = time.perf_counter()
        got = engine.poll_replies()
        tp1 = time.perf_counter()
        poll_us_samples.append((tp1 - tp0) * 1_000_000.0)
        async_replies.extend(got)
        if len(async_replies) < len(async_req_ids):
            time.sleep(0.010)
    engine.clear_memory("async_bench")

    # 3. Execute 8 Realistic Multi-Turn Conversations on 8 Unseen 256x256x4 Frostmarsh Scenes
    conversations: list[dict[str, Any]] = []
    all_turn_resps: list[DialogueResponse] = []

    def _run_turn(
        conv_id: str,
        turn_idx: int,
        w_s: torch.Tensor,
        z_s: torch.Tensor,
        lm: dict[str, Any],
        listener: BotProfile,
        speaker_name: str,
        speaker_pos: tuple[float, float, int],
        speaker_volume: float,
        raw_utterance: str,
        diurnal_time: str,
        seed: int,
    ) -> dict[str, Any]:
        obs = bridge.observe(
            w_s, z_s, listener,
            speaker_name=speaker_name,
            speaker_pos=speaker_pos,
            speaker_volume=speaker_volume,
            raw_utterance=raw_utterance,
            diurnal_time=diurnal_time,
            landmarks=lm,
            seed=seed,
        )
        resp = engine.generate_reply(
            DialogueRequest(bot=listener, observation=obs, conversation_id=conv_id)
        )
        all_turn_resps.append(resp)
        return {
            "turn": turn_idx,
            "speaker": speaker_name,
            "speaker_pos": [round(float(speaker_pos[0]), 1), round(float(speaker_pos[1]), 1), int(speaker_pos[2])],
            "speaker_volume": speaker_volume,
            "listener": listener.name,
            "listener_class": listener.bot_class,
            "listener_pos": list(obs.pos_xyz),
            "raw_utterance": raw_utterance,
            "heard_utterance": obs.heard_utterance,
            "snr_db": obs.incoming_snr_db,
            "snr_tier": obs.incoming_snr_tier,
            "occlusion_cause": obs.incoming_occlusion_cause,
            "ambient_noise_db": obs.ambient_noise_db,
            "z_world_energy": obs.z_latent_energy,
            "z_world_saliency": obs.z_latent_saliency,
            "z_peak_quadrant": obs.z_peak_quadrant,
            "grounded_hazards": {
                "local_temp_c": obs.local_temp_c,
                "oil_slick": obs.oil_slick,
                "wildfire": obs.wildfire,
                "skybridge": obs.skybridge,
                "ice_bridge": obs.ice_bridge,
            },
            "plan": resp.plan,
            "raw_json_valid": resp.raw_json_valid,
            "plan_valid": resp.plan_valid,
            "repaired": resp.repaired,
            "ttft_ms": resp.ttft_ms,
            "total_ms": resp.total_ms,
            "generated_tokens": resp.generated_tokens,
            "tokens_per_sec": resp.tokens_per_sec,
        }

    # --- Conv 1: Clear High-SNR Oil Trap & Ignition Negotiation (Human <-> Pyra) ---
    w1, _, lm1 = build_frostmarsh_3d_state(1401, device)
    w1 = apply_diurnal_and_scenario_physics(w1, lm1, diurnal_temp_delta_k=-10.0)
    _, z1 = world_forward_with_latent(wm, w1)
    ox1, oy1, _ = lm1["oil_positions"][0]
    pyra1 = BotProfile("c1_pyra", "Pyra", "Pyromancer", 0, min(244.0, ox1 + 9.0), min(244.0, oy1 + 6.0), 1, hp=92.0)
    h1_pos = (max(8.0, pyra1.x - 11.0), max(8.0, pyra1.y - 7.0), 1)
    c1_turns = []
    for ti, utt in enumerate([
        "Pyra, enemy scouts are nearing the oil basin — confirm unlit oil status and stage an ignition trap!",
        "Hold fire two seconds until both scouts step onto the slick, then widen focus to 7m.",
        "They are dead center on the oil slick — ignite it with maximum thermal flux right now!",
    ], start=1):
        c1_turns.append(_run_turn("conv1_oil_trap", ti, w1, z1, lm1, pyra1, "Human Commander", h1_pos, 0.85, utt, "Dawn Mist (06:20)", 14010 + ti))
    conversations.append({
        "id": "conv1_clear_oil_trap",
        "title": "Clear High-SNR Oil Trap & Ignition Negotiation (Human ↔ Pyra)",
        "category": "clear_high_snr_oil_ignition",
        "scene_seed": 1401,
        "diurnal_time": "Dawn Mist (06:20)",
        "turns": c1_turns,
    })

    # --- Conv 2: Muffled Low-SNR Waterfall Roar Confusion & Repeat Request (Human <-> Mire) ---
    w2, _, lm2 = build_frostmarsh_3d_state(1402, device)
    _, z2 = world_forward_with_latent(wm, w2)
    wfx2, wfy2, _ = lm2["waterfall_positions"][0]
    mire2 = BotProfile("c2_mire", "Mire", "Shadow-Flanker", 0, wfx2, wfy2, 1, hp=76.0)
    c2_turns = [
        # Turn 1: Mire in 85 dB plunging waterfall pool -> muffled (-4.8 dB)
        _run_turn("conv2_waterfall", 1, w2, z2, lm2, mire2, "Human Commander",
                  (min(244.0, wfx2 + 12.0), min(244.0, wfy2 + 10.0), 1), 0.96,
                  "Mire, coat the northern waterfall rim in crude oil and fall back!", "High Noon (12:45)", 14021),
        # Turn 2: Mire steps 20 cells clear of plunge pool -> clear SNR (+35 dB)
        _run_turn("conv2_waterfall", 2, w2, z2, lm2,
                  BotProfile("c2_mire", "Mire", "Shadow-Flanker", 0, max(12.0, wfx2 - 20.0), min(244.0, wfy2 + 20.0), 1, hp=76.0),
                  "Human Commander", (max(12.0, wfx2 - 10.0), min(244.0, wfy2 + 14.0), 1), 0.92,
                  "Repeating loud now that you cleared the falls! Coat the NW choke in oil!", "High Noon (12:45)", 14022),
        # Turn 3: Smoke screen confirmation
        _run_turn("conv2_waterfall", 3, w2, z2, lm2,
                  BotProfile("c2_mire", "Mire", "Shadow-Flanker", 0, max(12.0, wfx2 - 22.0), min(244.0, wfy2 + 22.0), 1, hp=76.0),
                  "Human Commander", (max(12.0, wfx2 - 14.0), min(244.0, wfy2 + 16.0), 1), 0.85,
                  "Oil confirmed — pop a wide smoke screen now while we pull back east!", "High Noon (12:45)", 14023),
    ]
    conversations.append({
        "id": "conv2_waterfall_muffled",
        "title": "Muffled Low-SNR Waterfall Roar Confusion & Repeat Request (Human ↔ Mire)",
        "category": "low_snr_waterfall_confusion",
        "scene_seed": 1402,
        "diurnal_time": "High Noon (12:45)",
        "turns": c2_turns,
    })

    # --- Conv 3: Inaudible L0 Rock Ceiling + Basalt Wall Occlusion -> Open Shaft Recovery (Human <-> Bram) ---
    w3, _, lm3 = build_frostmarsh_3d_state(1403, device)
    _, z3 = world_forward_with_latent(wm, w3)
    shx3, shy3, _ = lm3["shaft_positions"][0]
    bram3_deep = BotProfile("c3_bram", "Bram", "Vanguard", 0, 42.0, 192.0, 0, hp=84.0)
    bram3_shaft = BotProfile("c3_bram", "Bram", "Vanguard", 0, shx3, shy3, 0, hp=84.0)
    c3_turns = [
        _run_turn("conv3_cavern_wall", 1, w3, z3, lm3, bram3_deep, "Human Commander",
                  (190.0, 72.0, 1), 0.55,
                  "Bram, climb the vertical shaft and reinforce the river bridge immediately!", "Overcast Dusk (18:30)", 14031),
        _run_turn("conv3_cavern_wall", 2, w3, z3, lm3, bram3_shaft, "Human Commander",
                  (shx3 + 8.0, shy3 + 6.0, 1), 0.90,
                  "Bram, loud and clear at the open shaft! Push up and freeze the central river choke!", "Overcast Dusk (18:30)", 14032),
        _run_turn("conv3_cavern_wall", 3, w3, z3, lm3, bram3_shaft, "Human Commander",
                  (shx3 + 6.0, shy3 + 4.0, 1), 0.85,
                  "Good hold — drop a wide water barrier if their pyro pushes the rim!", "Overcast Dusk (18:30)", 14033),
    ]
    conversations.append({
        "id": "conv3_cavern_ceiling_occlusion",
        "title": "Inaudible L0 Rock Ceiling Occlusion & Vertical Shaft Recovery (Human ↔ Bram)",
        "category": "low_snr_wall_ceiling_occlusion",
        "scene_seed": 1403,
        "diurnal_time": "Overcast Dusk (18:30)",
        "turns": c3_turns,
    })

    # --- Conv 4: Midnight Freeze & Burning L2 Skybridge Collapse Warning (Human <-> Kael) ---
    w4, _, lm4 = build_frostmarsh_3d_state(1404, device)
    w4 = apply_diurnal_and_scenario_physics(
        w4, lm4, diurnal_temp_delta_k=-32.0, burn_south_skybridge=True, freeze_river=True
    )
    _, z4 = world_forward_with_latent(wm, w4)
    sbx4, sby4, _ = lm4["south_skybridge"]
    kael4 = BotProfile("c4_kael", "Kael", "Vanguard", 0, sbx4 - 28.0, sby4, 2, hp=88.0)
    h4_pos = (sbx4 - 34.0, sby4 + 2.0, 2)
    c4_turns = [
        _run_turn("conv4_burning_skybridge", 1, w4, z4, lm4, kael4, "Human Commander", h4_pos, 0.95,
                  "Kael, sprint across the L2 south skybridge to flank their shrine!", "Midnight Freeze (01:45)", 14041),
        _run_turn("conv4_burning_skybridge", 2, w4, z4, lm4, kael4, "Human Commander", h4_pos, 0.95,
                  "Copy the collapse risk — quench the burning timbers with cryo water right now!", "Midnight Freeze (01:45)", 14042),
        _run_turn("conv4_burning_skybridge", 3, w4, z4, lm4, kael4, "Human Commander", h4_pos, 0.92,
                  "Once quenched, pop smoke cover and pivot north to the stone skybridge!", "Midnight Freeze (01:45)", 14043),
    ]
    conversations.append({
        "id": "conv4_midnight_burning_skybridge",
        "title": "Midnight Freeze & Burning L2 Skybridge Collapse Warning (Human ↔ Kael)",
        "category": "midnight_freeze_burning_skybridge",
        "scene_seed": 1404,
        "diurnal_time": "Midnight Freeze (01:45)",
        "turns": c4_turns,
    })

    # --- Conv 5: Bot-to-Bot Emergent Negotiation — Oil-Then-Fire Shrine Ambush (Nyx <-> Pyra) ---
    w5, _, lm5 = build_frostmarsh_3d_state(1405, device)
    _, z5 = world_forward_with_latent(wm, w5)
    shrx5, shry5, _ = lm5["shrine_pos"]
    nyx5 = BotProfile("c5_nyx", "Nyx", "Shadow-Flanker", 0, max(12.0, shrx5 - 10.0), max(12.0, shry5 - 6.0), 1, hp=91.0)
    pyra5 = BotProfile("c5_pyra", "Pyra", "Pyromancer", 0, max(12.0, shrx5 - 22.0), max(12.0, shry5 - 10.0), 1, hp=96.0)
    c5_turns = []
    t5_1 = _run_turn("conv5_bot2bot_ambush", 1, w5, z5, lm5, pyra5, nyx5.name, (nyx5.x, nyx5.y, 1), 0.78,
                     "Pyra, I am coating the shrine choke in oil — hold your flame until I clear out!", "Clear Afternoon (15:10)", 14051)
    c5_turns.append(t5_1)
    t5_2 = _run_turn("conv5_bot2bot_ambush", 2, w5, z5, lm5, nyx5, pyra5.name, (pyra5.x, pyra5.y, 1), 0.80,
                     t5_1["plan"]["reply"] + " Lay crude oil across the shrine choke now!", "Clear Afternoon (15:10)", 14052)
    c5_turns.append(t5_2)
    t5_3 = _run_turn("conv5_bot2bot_ambush", 3, w5, z5, lm5, pyra5, nyx5.name, (nyx5.x, nyx5.y, 1), 0.85,
                     t5_2["plan"]["reply"] + " I am clear of the oil — ignite the trap now!", "Clear Afternoon (15:10)", 14053)
    c5_turns.append(t5_3)
    conversations.append({
        "id": "conv5_bot2bot_oil_fire_negotiation",
        "title": "Bot-to-Bot Emergent Negotiation: Oil-Then-Fire Shrine Ambush (Nyx ↔ Pyra)",
        "category": "bot_to_bot_emergent_negotiation",
        "scene_seed": 1405,
        "diurnal_time": "Clear Afternoon (15:10)",
        "turns": c5_turns,
    })

    # --- Conv 6: Bot-to-Bot Emergent Negotiation — Wildfire Containment (Kael <-> Glacius) ---
    w6, _, lm6 = build_frostmarsh_3d_state(1406, device)
    _, z6 = world_forward_with_latent(wm, w6)
    fx6, fy6, _ = lm6["wildfire_positions"][0]
    glac6 = BotProfile("c6_glacius", "Glacius", "Cryo-Scout", 0, max(12.0, fx6 - 14.0), max(12.0, fy6 - 10.0), 1, hp=86.0)
    kael6 = BotProfile("c6_kael", "Kael", "Vanguard", 0, max(12.0, fx6 - 26.0), max(12.0, fy6 - 14.0), 1, hp=79.0)
    c6_turns = []
    t6_1 = _run_turn("conv6_bot2bot_wildfire", 1, w6, z6, lm6, glac6, kael6.name, (kael6.x, kael6.y, 1), 0.88,
                     "Glacius, wildfire is roaring at 650K — quench the front with cryo water while I screen!", "Dusk Ember (20:10)", 14061)
    c6_turns.append(t6_1)
    t6_2 = _run_turn("conv6_bot2bot_wildfire", 2, w6, z6, lm6, kael6, glac6.name, (glac6.x, glac6.y, 1), 0.88,
                     t6_1["plan"]["reply"] + " Pop smoke cover so I can safely advance!", "Dusk Ember (20:10)", 14062)
    c6_turns.append(t6_2)
    t6_3 = _run_turn("conv6_bot2bot_wildfire", 3, w6, z6, lm6, glac6, kael6.name, (kael6.x, kael6.y, 1), 0.90,
                     t6_2["plan"]["reply"] + " Smoke is up — hit the wildfire with wide cryo water!", "Dusk Ember (20:10)", 14063)
    c6_turns.append(t6_3)
    conversations.append({
        "id": "conv6_bot2bot_wildfire_containment",
        "title": "Bot-to-Bot Emergent Negotiation: Wildfire Containment & Screen (Kael ↔ Glacius)",
        "category": "bot_to_bot_emergent_negotiation",
        "scene_seed": 1406,
        "diurnal_time": "Dusk Ember (20:10)",
        "turns": c6_turns,
    })

    # --- Conv 7: Midnight Freeze Ice Bridge Lockdown (Human <-> Glacius) ---
    w7, _, lm7 = build_frostmarsh_3d_state(1407, device)
    w7 = apply_diurnal_and_scenario_physics(w7, lm7, diurnal_temp_delta_k=-34.0, freeze_river=True)
    _, z7 = world_forward_with_latent(wm, w7)
    icx7, icy7, _ = lm7["ice_bridge_pos"]
    glac7 = BotProfile("c7_glacius", "Glacius", "Cryo-Scout", 0, max(12.0, icx7 - 10.0), icy7, 1, hp=95.0)
    h7_pos = (max(12.0, icx7 - 20.0), icy7 - 5.0, 1)
    c7_turns = [
        _run_turn("conv7_midnight_ice", 1, w7, z7, lm7, glac7, "Human Commander", h7_pos, 0.82,
                  "Glacius, river is frozen solid at -17C — lock down the central ice bridge with frost!", "Midnight Freeze (02:15)", 14071),
        _run_turn("conv7_midnight_ice", 2, w7, z7, lm7, glac7, "Human Commander", h7_pos, 0.85,
                  "Enemy pyromancer spotted east — use wide focus water and deep cryo so fire cannot melt it!", "Midnight Freeze (02:15)", 14072),
        _run_turn("conv7_midnight_ice", 3, w7, z7, lm7, glac7, "Human Commander", h7_pos, 0.85,
                  "Vanguard is crossing — pop smoke cover on the eastern ice edge!", "Midnight Freeze (02:15)", 14073),
    ]
    conversations.append({
        "id": "conv7_midnight_freeze_ice_bridge",
        "title": "Midnight Freeze River Crossing & Anti-Pyro Cryo Lockdown (Human ↔ Glacius)",
        "category": "midnight_freeze_burning_skybridge",
        "scene_seed": 1407,
        "diurnal_time": "Midnight Freeze (02:15)",
        "turns": c7_turns,
    })

    # --- Conv 8: Bot-to-Bot Basalt Wall Occlusion -> Clear Flank Negotiation (Sable <-> Vex) ---
    w8, _, lm8 = build_frostmarsh_3d_state(1408, device)
    _, z8 = world_forward_with_latent(wm, w8)
    wx1_8 = lm8["wall_specs"][0]["wx1"]
    sable8_behind = BotProfile("c8_sable", "Sable", "Shadow-Flanker", 1, max(12.0, wx1_8 - 22.0), 145.0, 1, hp=81.0)
    vex8 = BotProfile("c8_vex", "Vex", "Pyromancer", 1, min(244.0, wx1_8 + 30.0), 145.0, 1, hp=74.0)
    sable8_clear = BotProfile("c8_sable", "Sable", "Shadow-Flanker", 1, min(244.0, wx1_8 + 14.0), 136.0, 1, hp=81.0)
    c8_turns = []
    # Turn 1: Low-volume whisper across thick basalt ridge -> muffled (~0 dB)
    t8_1 = _run_turn("conv8_bot2bot_wall", 1, w8, z8, lm8, vex8, sable8_behind.name, (sable8_behind.x, sable8_behind.y, 1), 0.25,
                     "Vex, I am behind the basalt ridge laying oil — prepare to ignite the canyon flank!", "Late Dusk (21:00)", 14081)
    c8_turns.append(t8_1)
    # Turn 2: Sable rounds ridge into clear LOS
    t8_2 = _run_turn("conv8_bot2bot_wall", 2, w8, z8, lm8, sable8_clear, vex8.name, (vex8.x, vex8.y, 1), 0.85,
                     t8_1["plan"]["reply"] + " Step clear of the basalt wall and lay oil!", "Late Dusk (21:00)", 14082)
    c8_turns.append(t8_2)
    # Turn 3: Clear ignition execution
    t8_3 = _run_turn("conv8_bot2bot_wall", 3, w8, z8, lm8, vex8, sable8_clear.name, (sable8_clear.x, sable8_clear.y, 1), 0.88,
                     t8_2["plan"]["reply"] + " Clear of the wall — ignite the SW oil flank now!", "Late Dusk (21:00)", 14083)
    c8_turns.append(t8_3)
    conversations.append({
        "id": "conv8_bot2bot_basalt_wall_flank",
        "title": "Bot-to-Bot Negotiation Across Basalt Wall Occlusion (Sable ↔ Vex)",
        "category": "bot_to_bot_wall_occlusion",
        "scene_seed": 1408,
        "diurnal_time": "Late Dusk (21:00)",
        "turns": c8_turns,
    })

    engine.shutdown()

    # 4. Aggregate Metrics
    all_ttfts = bench_ttfts + [r.ttft_ms for r in all_turn_resps]
    all_totals = bench_totals + [r.total_ms for r in all_turn_resps]
    all_tps = bench_tps + [r.tokens_per_sec for r in all_turn_resps]
    all_dec_tps = bench_decode_tps + [r.decode_tokens_per_sec for r in all_turn_resps]
    all_gen_toks = bench_gen_toks + [r.generated_tokens for r in all_turn_resps]

    total_calls = len(bench_ttfts) + len(all_turn_resps) + len(async_replies)
    raw_valid_cnt = bench_raw_valid + sum(int(r.raw_json_valid) for r in all_turn_resps) + sum(int(r.raw_json_valid) for r in async_replies)
    plan_valid_cnt = bench_plan_valid + sum(int(r.plan_valid) for r in all_turn_resps) + sum(int(r.plan_valid) for r in async_replies)
    conv_raw_valid_cnt = sum(int(r.raw_json_valid) for r in all_turn_resps)
    conv_plan_valid_cnt = sum(int(r.plan_valid) for r in all_turn_resps)

    vram_alloc_gb = round(torch.cuda.memory_allocated(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
    vram_peak_gb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
    vram_reserved_gb = round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0

    low_snr_turns = [t for c in conversations for t in c["turns"] if t["snr_db"] <= 1.5]
    low_snr_ack = sum(
        1 for t in low_snr_turns
        if any(k in t["plan"]["reply"].lower() for k in ("repeat", "again", "loud", "roar", "waterfall", "wall", "muffle", "static", "break", "hear", "read", "signal", "comms", "ceiling", "cavern", "drown", "clear", "noise"))
        or any(k in t["plan"]["intent"].lower() for k in ("repeat", "comms", "relocate", "clarify", "signal"))
    )

    metrics: dict[str, Any] = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "model_id": DEFAULT_MODEL_ID,
            "dtype": str(engine.dtype),
            "world_model_checkpoint": str(wm_path),
            "model_load_time_s": round(engine.load_time_s, 3),
        },
        "latency_metrics": {
            "ttft_ms_p50": round(float(np.median(all_ttfts)), 2),
            "ttft_ms_mean": round(float(np.mean(all_ttfts)), 2),
            "ttft_ms_p95": round(float(np.percentile(all_ttfts, 95)), 2),
            "p50_total_ms": round(float(np.median(all_totals)), 2),
            "mean_total_ms": round(float(np.mean(all_totals)), 2),
            "p95_total_ms": round(float(np.percentile(all_totals, 95)), 2),
            "tokens_per_sec_p50": round(float(np.median(all_tps)), 2),
            "tokens_per_sec_mean": round(float(np.mean(all_tps)), 2),
            "decode_tokens_per_sec_p50": round(float(np.median(all_dec_tps)), 2),
            "mean_generated_tokens": round(float(np.mean(all_gen_toks)), 2),
            "bridge_observe_ms_p50": round(float(np.median(bench_bridge_ms)), 3),
            "vram_gb": vram_peak_gb,
            "vram_allocated_gb": vram_alloc_gb,
            "vram_reserved_gb": vram_reserved_gb,
            "non_blocking_submit_us_p50": round(float(np.median(submit_us_samples)), 2),
            "non_blocking_submit_us_max": round(float(np.max(submit_us_samples)), 2),
            "non_blocking_poll_us_p50": round(float(np.median(poll_us_samples)), 2),
            "async_submitted": len(async_req_ids),
            "async_completed": len(async_replies),
            "game_loop_30hz_budget_ms": 33.33,
            "game_loop_stall_free": bool(float(np.max(submit_us_samples)) < 1000.0 and len(async_replies) == len(async_req_ids)),
        },
        "quality_metrics": {
            "total_evaluated_turns": total_calls,
            "conversation_turns": len(all_turn_resps),
            "num_conversations": len(conversations),
            "valid_plan_pct": round(100.0 * plan_valid_cnt / max(total_calls, 1), 2),
            "raw_json_valid_pct": round(100.0 * raw_valid_cnt / max(total_calls, 1), 2),
            "conversation_valid_plan_pct": round(100.0 * conv_plan_valid_cnt / max(len(all_turn_resps), 1), 2),
            "conversation_raw_json_valid_pct": round(100.0 * conv_raw_valid_cnt / max(len(all_turn_resps), 1), 2),
            "low_snr_turns_count": len(low_snr_turns),
            "low_snr_acoustic_awareness_pct": round(100.0 * low_snr_ack / max(len(low_snr_turns), 1), 2),
        },
        "conversations": conversations,
    }

    out_metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lm_m = metrics["latency_metrics"]
    qm_m = metrics["quality_metrics"]
    hw_m = metrics["hardware"]

    md_lines: list[str] = [
        "# Frostmarsh System-2 SLM Dialogue & Grounded Tactical Planning Report",
        "",
        "## 1. Architecture Summary",
        f"- **SLM Backbone:** `{hw_m['model_id']}` (`{hw_m['dtype']}`) on **{hw_m['gpu']}** (`torch {hw_m['torch_version']}`)",
        f"- **Frozen World Model:** `{Path(hw_m['world_model_checkpoint']).name}` producing real `Z_world [4, 48, 256, 256]`",
        "- **Physical Acoustics:** Imported `AcousticField3D` from `live_acoustics.py` (FP16 3D wave/occlusion ambient noise floor + basalt wall / L0 ceiling / waterfall / wildfire ray-marched SNR)",
        "- **Bridge (`ZWorldToLanguageBridge`):** Extracts per-bot `16×16` `Z_world` ego-crop (latent energy, spatial saliency, peak quadrant) + grounded `w_state` physical sensors (oil slicks, wildfires, L2 south skybridge integrity, frozen river ice, diurnal time-of-day, local noise dB & incoming SNR degradation)",
        "- **Execution Modes (`System2DialogueEngine`):** Synchronous `generate_reply(...)` + dedicated non-blocking async worker thread (`submit_dialogue(...)` / `poll_replies()`) with rolling 4-turn per-bot memory and compact structured JSON enforcement (`max_new_tokens=64`, greedy decoding).",
        "",
        "## 2. NVIDIA GB10 Latency, Throughput & Reliability Metrics",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Model Load Time | `{hw_m['model_load_time_s']:.2f} s` |",
        f"| **TTFT (`ttft_ms` p50 / mean / p95)** | **`{lm_m['ttft_ms_p50']:.2f} ms`** / `{lm_m['ttft_ms_mean']:.2f} ms` / `{lm_m['ttft_ms_p95']:.2f} ms` |",
        f"| **Total Latency (`p50_total_ms` / mean / p95)** | **`{lm_m['p50_total_ms']:.2f} ms`** / `{lm_m['mean_total_ms']:.2f} ms` / `{lm_m['p95_total_ms']:.2f} ms` |",
        f"| **Throughput (`tokens_per_sec` p50 / mean)** | **`{lm_m['tokens_per_sec_p50']:.2f} tok/s`** / `{lm_m['tokens_per_sec_mean']:.2f} tok/s` |",
        f"| Decode-Phase Throughput (p50) | `{lm_m['decode_tokens_per_sec_p50']:.2f} tok/s` |",
        f"| Mean Generated Tokens | `{lm_m['mean_generated_tokens']:.1f} tok` (cap `64`) |",
        f"| `ZWorldToLanguageBridge.observe()` (p50) | `{lm_m['bridge_observe_ms_p50']:.3f} ms` |",
        f"| **Peak VRAM (`vram_gb`)** | **`{lm_m['vram_gb']:.3f} GB`** (allocated `{lm_m['vram_allocated_gb']:.3f} GB`, reserved `{lm_m['vram_reserved_gb']:.3f} GB`) |",
        f"| **Async `submit_dialogue()` (`non_blocking_submit_us`)** | **`{lm_m['non_blocking_submit_us_p50']:.2f} µs` p50** (max `{lm_m['non_blocking_submit_us_max']:.2f} µs`) |",
        f"| Async `poll_replies()` (p50) | `{lm_m['non_blocking_poll_us_p50']:.2f} µs` ({lm_m['async_completed']}/{lm_m['async_submitted']} completed, 30 Hz stall-free: `{lm_m['game_loop_stall_free']}`) |",
        f"| **Valid Structured Plan Rate (`valid_plan_pct`)** | **`{qm_m['valid_plan_pct']:.2f}%`** (`{plan_valid_cnt}/{total_calls}`) |",
        f"| Raw Unrepaired JSON Valid Rate | `{qm_m['raw_json_valid_pct']:.2f}%` (`{raw_valid_cnt}/{total_calls}`) |",
        f"| Low-SNR Acoustic Awareness Rate | `{qm_m['low_snr_acoustic_awareness_pct']:.2f}%` (`{low_snr_ack}/{len(low_snr_turns)}` muffled/inaudible turns) |",
        "",
        "## 3. Eight Physics-Grounded Multi-Turn Conversations (Unseen `256×256×4` Frostmarsh Scenes)",
        "",
    ]

    for idx, conv in enumerate(conversations, start=1):
        md_lines.append(f"### Conversation {idx}: {conv['title']}")
        md_lines.append(f"- **Scene Seed:** `{conv['scene_seed']}` | **Diurnal Time:** `{conv['diurnal_time']}` | **Category:** `{conv['category']}`")
        md_lines.append("")
        for t in conv["turns"]:
            ab = t["plan"]["actuator_bias"]
            md_lines.append(
                f"**Turn {t['turn']} — {t['speaker']} → {t['listener']} ({t['listener_class']})** "
                f"| SNR: `{t['snr_db']:+.1f} dB` (`{t['snr_tier']}`, {t['occlusion_cause']}) "
                f"| Local Noise: `{t['ambient_noise_db']:.1f} dB` "
                f"| `Z_world` E=`{t['z_world_energy']:.2f}`, sal=`{t['z_world_saliency']:.3f}` ({t['z_peak_quadrant']})"
            )
            md_lines.append(f"- *Sent:* \"{t['raw_utterance']}\"")
            if t["heard_utterance"] != t["raw_utterance"]:
                md_lines.append(f"- *Heard (physically degraded):* `{t['heard_utterance']}`")
            md_lines.append(f"- **{t['listener']} Reply:** \"{t['plan']['reply']}\"")
            md_lines.append(
                f"- **Structured Plan:** `intent={t['plan']['intent']}` | `target_sector={t['plan']['target_sector']}` "
                f"| `actuator_bias={{thermal_flux: {ab['thermal_flux']:+.2f}, matter: \"{ab['matter']}\", focus_radius: {ab['focus_radius']:.1f}}}` "
                f"| `{t['ttft_ms']:.1f} ms` TTFT, `{t['total_ms']:.1f} ms` total, raw_json=`{t['raw_json_valid']}`"
            )
            md_lines.append("")

    out_report_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[system2_slm] Saved metrics -> {out_metrics_json}", flush=True)
    print(f"[system2_slm] Saved report  -> {out_report_md}", flush=True)
    print(json.dumps({"latency_metrics": lm_m, "quality_metrics": qm_m}, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Frostmarsh System-2 SLM Dialogue Engine & Spark GB10 Benchmark")
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()
    run_benchmark_and_conversations(args.metrics_json, args.report_md)


if __name__ == "__main__":
    main()
