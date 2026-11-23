#!/usr/bin/env python3
"""
Production 3D Physical Acoustics, Emergent Bot Speech, Viewer Text Interpreter,
and Human-to-Bot Acoustic Command Override Engine (`LiveAcousticEngine`).

Designed for 10 Hz live execution (<1.5 ms on NVIDIA GB10) over dynamic Frostmarsh
3D worlds of arbitrary resolution (both 256x256x4 and 512x512x4 = 1,048,576 voxels!):

  1. `AcousticField3D`:
     - True FP16 10-step (8 coarse 64x64x4 + 2 fine HxWx4) 3D wall-aware wave/occlusion
       conv using -60 dB shifted power scaling (zero FP16 overflow/underflow!).
     - Ambient noise floor N(x,y,z) from plunging waterfalls (L2->L1), roaring wildfires
       (T > 440K), and explosive spells, plus dominant noise cause attribution.
     - Batched GPU ray-marching + basalt/ice wall occlusion (-12 dB/wall) + flanking
       diffraction + vertical cave ceiling (-18 dB) / plateau slab (-14 dB) / open shaft
       (-2 dB) attenuation.
  2. Emergent Bot-to-Bot Neural Speech & Viewer Interpretation:
     - Selects 1-2 salient alive bots from live Z_world ego-crops + local world hazard
       and combat proximity.
     - Runs trained neural Spotter -> AcousticField3D SNR to listener_pos ->
       ViewerInterpreter -> grounded speech bubbles with exact physical SNR degradation:
         >= +10 dB : crisp tactical callout
         +2..+10 dB: partial interference
         -6..+2 dB : muffled fragments
         < -6 dB   : [inaudible — drowned out by <cause>]
  3. Human Chat -> Physical Acoustic Propagation -> `g_i` Bot Action Override:
     - Propagates human voice from (hx, hy, hlayer) with volume through AcousticField3D
       to every bot.
     - Encodes natural-language human command into 16-d neural goal vector `g_i` +
       tactical steering override (move_x, move_y, elem_override, ability_boost, reply_text)
       for friendly bots with snr_db > 0.0, muffled "Say again?!" replies for -6..0 dB,
       and eavesdrop detection for enemy bots with snr_db > +6 dB.

Checkpoint: `/home/gooning/.t3/scratch/frostmarsh-ml/acoustic_comm_live.pt`
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import WorldModel

ROOT = Path("/home/gooning/.t3/scratch/frostmarsh-ml")
DEFAULT_CKPT_PATH = ROOT / "acoustic_comm_live.pt"
DEFAULT_WORLD_PT = ROOT / "world_model.pt"

# Channel indices matching FIELD_CONTRACT / sim_teacher.py
S_WATER = 0
S_ICE = 1
S_STEAM = 2
S_SOIL = 3
S_ROCK = 4
S_WOOD = 5
S_PLANT = 6
S_SMOKE = 7
S_OIL = 8
S_ASH = 9
CH_TEMP = 10
CH_ELEV = 14
CH_WALL = 15
CH_FORCING = 16

EVENT_NAMES = [
    "OIL_SLICK_IGNITE",
    "WILDFIRE_FLANK",
    "FROZEN_RIVER_BRIDGE",
    "SHRINE_AMBUSH",
    "WATERFALL_CHOKE",
    "CAVERN_SHAFT_FLANK",
]
SECTOR_NAMES = ["NW Highland", "NE Skybridge", "SW Oil Basin", "SE Shrine Valley"]
ELEMENT_NAMES = ["Fire", "Frost", "Oil"]
TACTIC_NAMES = [
    "IGNITE_TRAP",
    "SUPPRESS_FLANK",
    "SHATTER_BRIDGE",
    "OIL_AMBUSH",
    "FREEZE_CHOKE",
    "CAVERN_DROP",
]
SPEAKER_CLASSES = ["Vanguard", "Pyromancer", "Cryo-Scout", "Shadow-Flanker"]
ABILITY_NAMES = ["dash", "melee", "cast_bolt", "carve_ice", "ability_e", "ult_r"]

EVENT_TO_ELEM = [0, 1, 0, 2, 1, 2]
EVENT_TO_TACTIC = [0, 1, 2, 3, 4, 5]

# Physical acoustic constants
SRC_SPL_MIN_DB = 56.0
SRC_SPL_SPAN_DB = 38.0
WALL_LOSS_DB = 12.0
CAVE_CEIL_LOSS_DB = 18.0
PLATEAU_SLAB_LOSS_DB = 14.0
OPEN_VERT_LOSS_DB = 2.0
ENEMY_DETECT_SNR_DB = 6.0

# -60 dB power shift keeps 28..95 dB cleanly inside FP16 range [6.3e-4, 3.16e3] with zero overflow!
_FP16_DB_OFFSET = 60.0
_LN10_OVER_10 = math.log(10.0) / 10.0
_10_OVER_LN10 = 10.0 / math.log(10.0)

CAUSE_DESCRIPTIONS = {
    0: "distance attenuation",
    1: "L2→L1 plunging waterfall roar",
    2: "roaring wildfire crackle",
    3: "explosive spell blast",
}

# =============================================================================
# 1. Production Human Tactical Command Taxonomy & Bank
# =============================================================================
# Each spec: (intent_name, target_x_norm, target_y_norm, elem_idx, ability_idx, mirror_team1_x, action_summary)
LIVE_HUMAN_INTENTS: list[tuple[str, float, float, int, int, bool, str]] = [
    ("IGNITE_OIL_SPIRE",      0.50, 0.50, 0, 2, False, "Igniting oil at Central Spire"),
    ("FREEZE_NORTH_BRIDGE",   0.50, 0.24, 1, 3, False, "Freezing North Bridge"),
    ("RETREAT_SANCTUARY",     0.14, 0.50, 1, 0, True,  "Retreating to Sanctuary"),
    ("IGNITE_OIL_SW",         0.20, 0.76, 0, 2, False, "Igniting SW Oil Basin"),
    ("EXTINGUISH_WILDFIRE_NW",0.22, 0.28, 1, 2, False, "Extinguishing NW Wildfire"),
    ("HOLD_WATERFALL_NE",     0.76, 0.22, 1, 4, False, "Holding NE Waterfall Choke"),
    ("AMBUSH_SHRINE_SE",      0.80, 0.76, 2, 4, False, "Setting Oil Ambush at SE Shrine"),
    ("BREACH_SKYBRIDGE_FIRE", 0.50, 0.68, 0, 5, False, "Breaching Skybridge with Fire"),
    ("FLANK_CAVERN_OIL",      0.48, 0.78, 2, 0, False, "Flanking via Cavern Shaft"),
    ("FREEZE_CENTRAL_RIVER",  0.50, 0.50, 1, 3, False, "Freezing Central River Crossing"),
]

LIVE_COMMAND_BANK: list[list[str]] = [
    # 0: IGNITE_OIL_SPIRE
    [
        "ignite oil at central spire",
        "burn the oil slick at the central spire",
        "light up the oil pool around central spire",
        "fire bolt the central spire oil now",
        "torch oil at the center spire",
        "detonate the central spire oil trap with fire",
        "spark the oil slick by central spire",
        "set central spire oil ablaze",
        "shoot fire into oil at center spire",
        "ignite central spire crude oil immediately",
        "blast central spire oil slick with flame",
        "burn center spire oil pool",
    ],
    # 1: FREEZE_NORTH_BRIDGE
    [
        "freeze north bridge",
        "freeze the north bridge with frost",
        "ice over the northern bridge crossing",
        "carve ice barrier across north bridge",
        "lock down the north bridge in frost",
        "frost wall on the northern bridge now",
        "chill the north bridge solid",
        "freeze the north river bridge",
        "seal northern bridge with ice",
        "frost bolt north bridge immediately",
        "turn north bridge into solid ice",
        "hold and freeze north bridge",
    ],
    # 2: RETREAT_SANCTUARY
    [
        "retreat to sanctuary",
        "fall back to sanctuary now",
        "pull back to our sanctuary",
        "dash back to sanctuary cover",
        "disengage and retreat to sanctuary",
        "regroup at sanctuary immediately",
        "withdraw to base sanctuary",
        "evade and fall back toward sanctuary",
        "retreat behind sanctuary ridge cover",
        "bail out back to sanctuary",
        "run back to team sanctuary",
        "fall back west to sanctuary base",
    ],
    # 3: IGNITE_OIL_SW
    [
        "ignite oil slick in southwest basin",
        "burn the southwest oil pool with fire",
        "light up the sw oil trap now",
        "fire bolt the southwest oil slick",
        "detonate sw oil puddle with flame",
        "torch the southwest crude oil basin",
        "spark sw oil field immediately",
        "set southwest basin oil ablaze",
        "cast fire on southwest oil slick",
        "burn oil down southwest",
    ],
    # 4: EXTINGUISH_WILDFIRE_NW
    [
        "extinguish wildfire on northwest ridge",
        "put out the nw forest fire with frost",
        "suppress northwest burning woods",
        "douse the roaring wildfire northwest",
        "cast frost on the nw blaze",
        "quench northwest forest flames",
        "stop the wildfire northwest using frost",
        "freeze out the nw timber fire",
        "cool down northwest wildfire front",
        "snuff out nw burning trees with frost",
    ],
    # 5: HOLD_WATERFALL_NE
    [
        "hold northeast waterfall choke with frost",
        "defend the ne waterfall high ground",
        "freeze northeast falls bottleneck",
        "guard ne highland waterfall rim",
        "lock down northeast waterfall cliff",
        "block ne waterfall passage with ice",
        "deploy frost barrier at northeast falls",
        "secure ne cliff waterfall choke",
        "stand firm on northeast falls",
        "hold the ne cascade choke",
    ],
    # 6: AMBUSH_SHRINE_SE
    [
        "lay oil ambush at southeast shrine",
        "coat the se shrine alcove in oil",
        "prepare oil trap by southeast shrine",
        "spill oil around se shrine choke",
        "ambush enemies at southeast shrine with oil",
        "slick the se shrine floor in oil",
        "flood southeast shrine entrance with oil",
        "stage oil trap at se shrine",
        "drench southeast shrine in crude oil",
        "set se shrine oil ambush now",
    ],
    # 7: BREACH_SKYBRIDGE_FIRE
    [
        "breach skybridge with fire ult",
        "push the south skybridge using flame",
        "blast the skybridge with fire ultimate",
        "storm the skybridge arch in flames",
        "burn through skybridge defense with fire",
        "fire ult across the skybridge now",
        "torch the high skybridge choke",
        "nuke the skybridge with flame ult",
        "assault skybridge firing flames",
        "heavy fire push on skybridge",
    ],
    # 8: FLANK_CAVERN_OIL
    [
        "drop into cavern shaft and flank",
        "sneak through underground tunnel with oil",
        "dash down cavern shaft silently",
        "flank via subterranean cave south",
        "quiet cavern shaft flanking maneuver",
        "slip into south cavern tunnel with oil",
        "stealth dash through underground shaft",
        "take lower cavern flank with oil",
        "enter tunnel shaft quietly",
        "underground cave shaft flank",
    ],
    # 9: FREEZE_CENTRAL_RIVER
    [
        "freeze central river crossing",
        "ice over the middle river ford",
        "carve ice bridge across central river",
        "solidify the center stream with frost",
        "freeze middle water crossing now",
        "chill central river into solid ice",
        "build frost crossing over center river",
        "seal the center river with ice",
        "frost across the middle stream",
        "freeze the central water ford",
    ],
]


def encode_text_hash_bow(
    texts: list[str],
    dim: int = 256,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Fast deterministic word + char-3gram hashed bag-of-features [B, 256]."""
    feats = np.zeros((len(texts), dim), dtype=np.float32)
    for i, txt in enumerate(texts):
        clean = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in txt.lower())
        words = clean.split()
        for w in words:
            h = int(hashlib.md5(f"W:{w}".encode()).hexdigest()[:8], 16) % dim
            feats[i, h] += 1.6
            padded = f"<{w}>"
            for j in range(max(1, len(padded) - 2)):
                tri = padded[j : j + 3]
                h3 = int(hashlib.md5(f"C:{tri}".encode()).hexdigest()[:8], 16) % dim
                feats[i, h3] += 0.45
        norm = float(np.linalg.norm(feats[i]))
        if norm > 1e-6:
            feats[i] /= norm
    return torch.from_numpy(feats).to(device=device, non_blocking=True)


# =============================================================================
# 2. Dynamic-Resolution FP16 3D Physical Acoustics Solver (`AcousticField3D`)
# =============================================================================
class AcousticField3D(nn.Module):
    """
    Dynamic-resolution GPU-accelerated 3D physical acoustics solver working natively
    for BOTH 256x256x4 and 512x512x4 (and any H, W divisible by 4).

    Uses -60 dB shifted linear power so the 10-step 3D wave/occlusion conv runs in
    pure IEEE FP16 with zero overflow/underflow in <0.5 ms on NVIDIA GB10!
    """

    def __init__(self, layers: int = 4, n_coarse_steps: int = 8, n_fine_steps: int = 2):
        super().__init__()
        self.layers = layers
        self.n_coarse_steps = n_coarse_steps
        self.n_fine_steps = n_fine_steps

        k2d = torch.tensor(
            [[0.06, 0.14, 0.06],
             [0.14, 0.20, 0.14],
             [0.06, 0.14, 0.06]],
            dtype=torch.float32,
        )
        k2d = k2d / k2d.sum()
        self.register_buffer("horiz_kernel_f32", k2d.view(1, 1, 3, 3).repeat(layers, 1, 1, 1))
        self.register_buffer("horiz_kernel_f16", k2d.view(1, 1, 3, 3).repeat(layers, 1, 1, 1).half())

        base_db = torch.tensor([28.0, 32.0, 34.0, 36.0], dtype=torch.float32).view(layers, 1, 1)
        self.register_buffer("base_noise_db", base_db)
        base_p_scaled = torch.exp((base_db - _FP16_DB_OFFSET) * _LN10_OVER_10)
        self.register_buffer("base_p_scaled_f32", base_p_scaled)
        self.register_buffer("base_p_scaled_f16", base_p_scaled.half())

        # Pre-cached 24-sample ray-march parameter vector for zero-alloc pairwise links
        t_ray = torch.linspace(0.06, 0.94, 24, dtype=torch.float32).view(1, 24)
        self.register_buffer("t_ray", t_ray)
        self.register_buffer("sin_t_ray", torch.sin(t_ray * math.pi))
        self.register_buffer("signs_2", torch.tensor([-1.0, 1.0], dtype=torch.float32).view(2, 1, 1))
        # Indices of the 8 channels read by AcousticField3D: water, ice, soil, rock, temp, elev, wall, forcing
        self.register_buffer(
            "ac_ch_idx",
            torch.tensor([S_WATER, S_ICE, S_SOIL, S_ROCK, CH_TEMP, CH_ELEV, CH_WALL, CH_FORCING], dtype=torch.long),
        )
        # Per-shape CUDA Graph cache for (extract_geometry + 10-step FP16 ambient noise + 64-link ray-march)
        self._noise_graphs: dict[tuple[int, int, int, bool], dict[str, Any]] = {}
        self._max_graph_links = 64

    def _geometry_and_noise_from_w8(
        self,
        w8_in: torch.Tensor,
        fp16: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core geometry + 10-step 3D wave/occlusion ambient noise from 8 acoustic channels [L, 8, H, W]."""
        dt = torch.float16 if fp16 else torch.float32
        L, _, H, W = w8_in.shape
        # 8 acoustic channels: 0:water, 1:ice, 2:soil, 3:rock, 4:temp, 5:elev, 6:wall, 7:forcing
        if H > 256 or W > 256:
            ds_h = max(1, H // 256)
            ds_w = max(1, W // 256)
            w8 = F.max_pool2d(w8_in, kernel_size=(ds_h, ds_w), stride=(ds_h, ds_w))
        else:
            w8 = w8_in
        Ha, Wa = w8.shape[2], w8.shape[3]

        solid_wall = ((w8[:, 3] >= 0.5) | (w8[:, 6] >= 0.5) | (w8[:, 1] >= 0.70)).to(dtype=dt)
        open_shaft_01 = ((w8[0, 3] < 0.5) & (w8[0, 5] > 0.55)).to(dtype=dt)
        open_sky_12 = ((w8[2, 2] < 0.30) & (w8[2, 3] < 0.50)).to(dtype=dt) if L > 2 else torch.ones_like(open_shaft_01)

        # Build physical noise sources on [L, Ha, Wa]
        p_wf = torch.zeros((1, L, Ha, Wa), device=w8.device, dtype=dt)
        if L >= 3:
            l2_water = (w8[2, 0] > 0.55).to(dtype=dt)
            l2_edge = (
                F.max_pool2d(open_sky_12.view(1, 1, Ha, Wa), 5, stride=1, padding=2).squeeze(0).squeeze(0) > 0.5
            ).to(dtype=dt)
            wf_lip_l2 = l2_water * l2_edge
            wf_plunge_l1 = (
                F.max_pool2d(wf_lip_l2.view(1, 1, Ha, Wa), 7, stride=1, padding=3).squeeze(0).squeeze(0)
                * (w8[1, 0] > 0.50).to(dtype=dt)
            )
            p_wf[0, 1] = wf_plunge_l1 * float(10.0 ** ((85.0 - _FP16_DB_OFFSET) / 10.0))
            p_wf[0, 2] = wf_lip_l2 * float(10.0 ** ((81.0 - _FP16_DB_OFFSET) / 10.0))

        hot = torch.clamp((w8[:, 4] - 1.5) * 0.4, 0.0, 1.0)
        p_fire = ((hot > 0.05).to(dtype=dt) * torch.exp(((70.0 - _FP16_DB_OFFSET) + 7.0 * hot) * _LN10_OVER_10)).unsqueeze(0)

        spell_mag = torch.clamp(w8[:, 7].abs() * 0.5, 0.0, 1.0)
        p_spell = (
            (spell_mag > 0.20).to(dtype=dt) * torch.exp(((84.0 - _FP16_DB_OFFSET) + 6.0 * spell_mag) * _LN10_OVER_10)
        ).unsqueeze(0)

        # 10-step 3D wave/occlusion conv:
        #   - 8 coarse 64x64xL steps with B=3 (separate wf/fire/spell for exact dominant cause attribution!)
        #   - 2 fine HaxWaxL steps with B=1 on summed linear power (exact by linearity, 3x faster!)
        kernel = self._get_kernel(L, dt)
        stride = max(4, min(Ha, Wa) // 64)
        wall_f = solid_wall.unsqueeze(0)
        trans_f = 1.0 - 0.90 * wall_f
        v_up_f, v_dn_f, v_self_f = self._build_vert_tensors(open_shaft_01, open_sky_12, layers=L)

        stacked = torch.cat([p_wf, p_fire, p_spell], dim=0)  # [3, L, Ha, Wa]
        p_c = F.max_pool2d(stacked, kernel_size=stride, stride=stride) * 0.55
        wall_c = F.max_pool2d(wall_f, kernel_size=stride, stride=stride)
        trans_c = 1.0 - 0.94 * wall_c
        v_up_c = F.avg_pool2d(v_up_f, kernel_size=stride, stride=stride)
        v_dn_c = F.avg_pool2d(v_dn_f, kernel_size=stride, stride=stride)
        v_self_c = 1.0 - v_up_c - v_dn_c

        fc = p_c
        for _ in range(self.n_coarse_steps):
            fc = self._step_3d_conv(fc, trans_c, v_up_c, v_dn_c, v_self_c, kernel, 0.52)
            fc = torch.maximum(fc, p_c)

        p_base = self.base_p_scaled_f16 if dt == torch.float16 else self.base_p_scaled_f32
        if L != self.layers:
            p_base = p_base[:L]

        # Dominant cause on coarse 64x64xL -> upsample nearest to HaxWa (saves 16x work!)
        p_base_c = p_base.expand_as(fc[0]) * 1.5
        cause_c = torch.stack([p_base_c, fc[0], fc[1], fc[2]], dim=0).argmax(dim=0).to(dtype=dt)
        noise_cause = F.interpolate(cause_c.unsqueeze(0), size=(Ha, Wa), mode="nearest").squeeze(0).to(dtype=torch.int8)

        # Sum linear acoustic power -> 2 fine 256x256xL refinement steps with B=1!
        p_sum_c = fc.sum(dim=0, keepdim=True)  # [1, L, 64, 64]
        p_sum_src = stacked.sum(dim=0, keepdim=True)  # [1, L, Ha, Wa]
        ff = F.interpolate(p_sum_c, size=(Ha, Wa), mode="bilinear", align_corners=False)
        ff = torch.maximum(ff, p_sum_src)
        for _ in range(self.n_fine_steps):
            ff = self._step_3d_conv(ff, trans_f, v_up_f, v_dn_f, v_self_f, kernel, 0.82)
            ff = torch.maximum(ff, p_sum_src)

        p_total = p_base + ff.squeeze(0)
        noise_db = _FP16_DB_OFFSET + _10_OVER_LN10 * torch.log(torch.clamp(p_total, min=1e-4))
        return solid_wall, open_shaft_01, open_sky_12, noise_db, noise_cause

    @torch.inference_mode()
    def compute_geometry_and_noise(
        self,
        w_state: torch.Tensor,
        use_fp16: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute (solid_wall, open_shaft_01, open_sky_12, noise_db, noise_cause) from w_state [L, 18, H, W].
        Captures and replays a per-(L, H, W) CUDA Graph on GPU (plus a companion 64-link ray-march graph
        bound directly to the same static VRAM buffers!) so 256x256x4 & 512x512x4 stay well under 1.5 ms!
        """
        L, C, H, W = w_state.shape
        device = w_state.device
        fp16 = bool(use_fp16) and (device.type == "cuda")
        dt = torch.float16 if fp16 else torch.float32

        if device.type != "cuda":
            w8 = w_state.index_select(1, self.ac_ch_idx).to(dtype=dt)
            return self._geometry_and_noise_from_w8(w8, fp16=False)

        key = (L, H, W, fp16)
        entry = self._noise_graphs.get(key)
        if entry is None:
            static_w8 = torch.zeros((L, 8, H, W), device=device, dtype=dt)
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    _ = self._geometry_and_noise_from_w8(static_w8, fp16=fp16)
            torch.cuda.current_stream(device).wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                g_sw, g_sh, g_sk, g_ndb, g_nc = self._geometry_and_noise_from_w8(static_w8, fp16=fp16)

            # Capture companion 64-link CUDA Graph bound directly to (g_sw, g_sh, g_sk, g_ndb, g_nc)
            M_cap = self._max_graph_links
            st_src = torch.zeros((M_cap, 3), device=device, dtype=torch.float32)
            st_dst = torch.ones((M_cap, 3), device=device, dtype=torch.float32)
            s2 = torch.cuda.Stream(device=device)
            s2.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(s2):
                for _ in range(3):
                    _ = self.query_pairwise_link(st_src, st_dst, g_sw, g_sh, g_sk, g_ndb, g_nc)
            torch.cuda.current_stream(device).wait_stream(s2)

            g_link = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_link):
                g_link_out = self.query_pairwise_link(st_src, st_dst, g_sw, g_sh, g_sk, g_ndb, g_nc)

            entry = {
                "graph": g,
                "static_w8": static_w8,
                "out": (g_sw, g_sh, g_sk, g_ndb, g_nc),
                "ac_hw": (int(g_sw.shape[1]), int(g_sw.shape[2])),
                "link_graph": g_link,
                "link_src": st_src,
                "link_dst": st_dst,
                "link_out": g_link_out,
            }
            self._noise_graphs[key] = entry

        torch.index_select(w_state.to(dtype=dt), 1, self.ac_ch_idx, out=entry["static_w8"])
        entry["graph"].replay()
        self._last_graph_key = key
        return entry["out"]

    @torch.inference_mode()
    def query_pairwise_link_fast(
        self,
        src_xyz: torch.Tensor,
        dst_xyz: torch.Tensor,
        world_hw: tuple[int, int],
        solid_wall: torch.Tensor,
        open_shaft_01: torch.Tensor,
        open_sky_12: torch.Tensor,
        noise_db: torch.Tensor,
        noise_cause: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        CUDA-graphed pairwise acoustic link query (~0.05 ms!) automatically scaling coordinates
        from world (H, W) into the acoustic grid (Ha, Wa). Falls back cleanly if M > 64 or on CPU.
        """
        H, W = world_hw
        Ha, Wa = solid_wall.shape[1], solid_wall.shape[2]
        if Ha != H or Wa != W:
            sx = float(Wa - 1) / float(max(W - 1, 1))
            sy = float(Ha - 1) / float(max(H - 1, 1))
            scale_t = torch.tensor([sx, sy, 1.0], device=src_xyz.device, dtype=torch.float32)
            src_q = src_xyz * scale_t
            dst_q = dst_xyz * scale_t
        else:
            src_q = src_xyz
            dst_q = dst_xyz

        M = src_q.shape[0]
        key = getattr(self, "_last_graph_key", None)
        entry = self._noise_graphs.get(key) if key is not None else None
        if (
            entry is not None
            and src_q.device.type == "cuda"
            and M <= self._max_graph_links
            and entry["out"][0].data_ptr() == solid_wall.data_ptr()
        ):
            entry["link_src"][:M].copy_(src_q, non_blocking=True)
            entry["link_dst"][:M].copy_(dst_q, non_blocking=True)
            entry["link_graph"].replay()
            return {k: v[:M] for k, v in entry["link_out"].items()}

        return self.query_pairwise_link(src_q, dst_q, solid_wall, open_shaft_01, open_sky_12, noise_db, noise_cause)

    def _get_kernel(self, layers: int, dtype: torch.dtype) -> torch.Tensor:
        k = self.horiz_kernel_f16 if dtype == torch.float16 else self.horiz_kernel_f32
        if layers == self.layers:
            return k
        return k[0:1].repeat(layers, 1, 1, 1)

    @staticmethod
    def extract_acoustic_geometry(
        w_state: torch.Tensor,
        open_shaft_override: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract solid wall mask [L, H, W], open L0<->L1 shaft [H, W], open L1<->L2 sky [H, W]."""
        out_dt = dtype if dtype is not None else w_state.dtype
        solid_wall = (
            (w_state[:, S_ROCK] >= 0.5)
            | (w_state[:, CH_WALL] >= 0.5)
            | (w_state[:, S_ICE] >= 0.70)
        ).to(dtype=out_dt)

        if open_shaft_override is not None:
            open_shaft_01 = open_shaft_override.to(device=w_state.device, dtype=out_dt)
        else:
            open_shaft_01 = ((w_state[0, S_ROCK] < 0.5) & (w_state[0, CH_ELEV] > 0.55)).to(dtype=out_dt)

        if w_state.shape[0] > 2:
            open_sky_12 = ((w_state[2, S_SOIL] < 0.30) & (w_state[2, S_ROCK] < 0.50)).to(dtype=out_dt)
        else:
            open_sky_12 = torch.ones_like(open_shaft_01)
        return solid_wall, open_shaft_01, open_sky_12

    def _build_vert_tensors(
        self,
        open_shaft_01: torch.Tensor,
        open_sky_12: torch.Tensor,
        layers: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized [1, L, H, W] vertical coupling coefficients for L0<->L1<->L2<->L3."""
        z0 = torch.zeros_like(open_shaft_01)
        c01 = 0.35 * (0.016 + 0.68 * open_shaft_01)
        c12 = 0.35 * (0.040 + 0.66 * open_sky_12)
        c23 = torch.full_like(open_shaft_01, 0.30 * 0.35)
        if layers == 4:
            v_up = torch.stack([z0, c01, c12, c23], dim=0).unsqueeze(0)
            v_dn = torch.stack([c01, c12, c23, z0], dim=0).unsqueeze(0)
        else:
            ups = [z0]
            dns = []
            for l in range(layers - 1):
                c = c01 if l == 0 else (c12 if l == 1 else c23)
                ups.append(c)
                dns.append(c)
            dns.append(z0)
            v_up = torch.stack(ups, dim=0).unsqueeze(0)
            v_dn = torch.stack(dns, dim=0).unsqueeze(0)
        v_self = 1.0 - v_up - v_dn
        return v_up, v_dn, v_self

    def _step_3d_conv(
        self,
        energy: torch.Tensor,
        trans_h: torch.Tensor,
        v_up: torch.Tensor,
        v_dn: torch.Tensor,
        v_self: torch.Tensor,
        kernel: torch.Tensor,
        damping: float,
    ) -> torch.Tensor:
        L = energy.shape[1]
        horiz = F.conv2d(energy * trans_h, kernel, padding=1, groups=L)
        e_from_below = F.pad(horiz[:, :-1], (0, 0, 0, 0, 1, 0))
        e_from_above = F.pad(horiz[:, 1:], (0, 0, 0, 0, 0, 1))
        return (horiz * v_self + e_from_below * v_up + e_from_above * v_dn) * damping

    def propagate_power_3d(
        self,
        src_power: torch.Tensor,      # [B, L, H, W] scaled linear acoustic power
        solid_wall: torch.Tensor,     # [L, H, W]
        open_shaft_01: torch.Tensor,  # [H, W]
        open_sky_12: torch.Tensor,    # [H, W]
        damping_coarse: float = 0.52,
        damping_fine: float = 0.82,
    ) -> torch.Tensor:
        """
        10-step (8 coarse 64x64xL + 2 fine HxWxL) GPU 3D damped wave/occlusion propagation.
        Automatically picks coarse stride (4 for 256x256, 8 for 512x512) so physical wavelength
        and GPU latency stay fast across both resolutions!
        """
        _, L, H, W = src_power.shape
        dt = src_power.dtype
        kernel = self._get_kernel(L, dt)

        # Coarse grid targets ~64x64 so 8 steps span the full arena identically at 256 and 512
        stride = max(4, min(H, W) // 64)
        while stride > 1 and ((H % stride != 0) or (W % stride != 0)):
            stride //= 2

        wall_f = solid_wall.to(dtype=dt).unsqueeze(0)  # [1, L, H, W]
        trans_f = 1.0 - 0.90 * wall_f
        v_up_f, v_dn_f, v_self_f = self._build_vert_tensors(
            open_shaft_01.to(dtype=dt), open_sky_12.to(dtype=dt), layers=L
        )

        # 8 coarse steps (max-pool wall so thin basalt ridges never leak sound!)
        p_c = F.max_pool2d(src_power, kernel_size=stride, stride=stride) * 0.55
        wall_c = F.max_pool2d(wall_f, kernel_size=stride, stride=stride)
        trans_c = 1.0 - 0.94 * wall_c
        v_up_c = F.avg_pool2d(v_up_f, kernel_size=stride, stride=stride)
        v_dn_c = F.avg_pool2d(v_dn_f, kernel_size=stride, stride=stride)
        v_self_c = 1.0 - v_up_c - v_dn_c

        field_c = p_c
        for _ in range(self.n_coarse_steps):
            field_c = self._step_3d_conv(field_c, trans_c, v_up_c, v_dn_c, v_self_c, kernel, damping_coarse)
            field_c = torch.maximum(field_c, p_c)

        # Upsample + 2 fine HxWxL steps (total = 10 3D wave/occlusion conv steps!)
        field_f = F.interpolate(field_c, size=(H, W), mode="bilinear", align_corners=False)
        field_f = torch.maximum(field_f, src_power)
        for _ in range(self.n_fine_steps):
            field_f = self._step_3d_conv(field_f, trans_f, v_up_f, v_dn_f, v_self_f, kernel, damping_fine)
            field_f = torch.maximum(field_f, src_power)
        return field_f

    @torch.inference_mode()
    def compute_ambient_noise_field(
        self,
        w_state: torch.Tensor,
        solid_wall: torch.Tensor,
        open_shaft_01: torch.Tensor,
        open_sky_12: torch.Tensor,
        use_fp16: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute authoritative 3D Ambient Noise Floor N(x,y,z) [L, H, W] (dB SPL)
        and dominant physical noise cause [L, H, W] (0=ambient, 1=waterfall, 2=wildfire, 3=spell).
        """
        device = w_state.device
        dt = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32
        L, _, H, W = w_state.shape
        w = w_state.to(dtype=dt)

        # 1. Plunging waterfalls (L2 water over open cliff edge + L1 plunge pool)
        p_wf = torch.zeros((1, L, H, W), device=device, dtype=dt)
        if L >= 3:
            l2_water = (w[2, S_WATER] > 0.55).to(dtype=dt)
            l2_edge = (
                F.max_pool2d(open_sky_12.to(dtype=dt).view(1, 1, H, W), 5, stride=1, padding=2).squeeze(0).squeeze(0)
                > 0.5
            ).to(dtype=dt)
            wf_lip_l2 = l2_water * l2_edge
            wf_plunge_l1 = (
                F.max_pool2d(wf_lip_l2.view(1, 1, H, W), 7, stride=1, padding=3).squeeze(0).squeeze(0)
                * (w[1, S_WATER] > 0.50).to(dtype=dt)
            )
            # Shifted by -60 dB: 85 dB -> 316.23, 81 dB -> 125.89 (exact FP16 safe!)
            p_wf[0, 1] = wf_plunge_l1 * float(10.0 ** ((85.0 - _FP16_DB_OFFSET) / 10.0))
            p_wf[0, 2] = wf_lip_l2 * float(10.0 ** ((81.0 - _FP16_DB_OFFSET) / 10.0))

        # 2. Roaring wildfires (T > 440 K => ch10 > 1.5 -> 70..77 dB)
        hot = torch.clamp((w[:, CH_TEMP] - 1.5) * 0.4, 0.0, 1.0)
        wf_fire_mask = (hot > 0.05).to(dtype=dt)
        fire_db_s = (70.0 - _FP16_DB_OFFSET) + 7.0 * hot
        p_fire = (wf_fire_mask * torch.exp(fire_db_s * _LN10_OVER_10)).unsqueeze(0)

        # 3. Explosive spells / rapid steam bursts (84..90 dB)
        spell_mag = torch.clamp(w[:, CH_FORCING].abs() * 0.5, 0.0, 1.0)
        spell_mask = (spell_mag > 0.20).to(dtype=dt)
        spell_db_s = (84.0 - _FP16_DB_OFFSET) + 6.0 * spell_mag
        p_spell = (spell_mask * torch.exp(spell_db_s * _LN10_OVER_10)).unsqueeze(0)

        # Batch-propagate all 3 physical noise fields in ONE 10-step FP16 3D conv pass!
        stacked = torch.cat([p_wf, p_fire, p_spell], dim=0)  # [3, L, H, W]
        prop = self.propagate_power_3d(stacked, solid_wall, open_shaft_01, open_sky_12)

        p_base = self.base_p_scaled_f16 if dt == torch.float16 else self.base_p_scaled_f32
        if L != self.layers:
            p_base = p_base[:L]
        p_total = p_base + prop[0] + prop[1] + prop[2]
        noise_db = _FP16_DB_OFFSET + _10_OVER_LN10 * torch.log(torch.clamp(p_total, min=1e-4))

        all_sources = torch.stack([p_base.expand_as(prop[0]) * 1.5, prop[0], prop[1], prop[2]], dim=0)
        noise_cause = all_sources.argmax(dim=0).to(dtype=torch.int8)
        return noise_db, noise_cause

    @torch.inference_mode()
    def query_pairwise_link(
        self,
        src_xyz: torch.Tensor,        # [M, 3] (x in [0..W-1], y in [0..H-1], z in [0..L-1])
        dst_xyz: torch.Tensor,        # [M, 3]
        solid_wall: torch.Tensor,     # [L, H, W]
        open_shaft_01: torch.Tensor,  # [H, W]
        open_sky_12: torch.Tensor,    # [H, W]
        noise_db: torch.Tensor,       # [L, H, W]
        noise_cause: torch.Tensor,    # [L, H, W]
    ) -> dict[str, torch.Tensor]:
        """
        Fast batched GPU query for M (src -> dst) acoustic links on arbitrary [L, H, W] worlds.
        Scales distance & diffraction proportionally to arena resolution so 256x256 and 512x512
        exhibit consistent physical SNR curves.
        """
        L, H, W = solid_wall.shape
        scale_to_256 = 256.0 / float(max(H, W))
        inv_scale = float(max(H, W)) / 256.0

        dx = dst_xyz[:, 0] - src_xyz[:, 0]
        dy = dst_xyz[:, 1] - src_xyz[:, 1]
        dz = dst_xyz[:, 2] - src_xyz[:, 2]

        # Canonical arena-unit distance (consistent across 256x256 and 512x512!)
        dx_m = dx * scale_to_256
        dy_m = dy * scale_to_256
        dist = torch.sqrt(dx_m * dx_m + dy_m * dy_m + (dz * 10.0) ** 2 + 1e-6)
        pl_geom = 20.0 * torch.log10(1.0 + dist * 0.25)

        t = self.t_ray
        rx = (src_xyz[:, 0:1] + t * dx.unsqueeze(1)).clamp(0, W - 1).long()
        ry = (src_xyz[:, 1:2] + t * dy.unsqueeze(1)).clamp(0, H - 1).long()
        rz = torch.round(src_xyz[:, 2:3] + t * dz.unsqueeze(1)).clamp(0, L - 1).long()

        w_s = solid_wall[rz, ry, rx].float()  # [M, 24]
        prev_w = F.pad(w_s[:, :-1], (1, 0), value=0.0)
        wall_entries = F.relu(w_s - prev_w).sum(dim=1)
        wall_thick = w_s.mean(dim=1) * (dist / 12.0).clamp(max=2.5)
        n_walls = torch.maximum(wall_entries, wall_thick * 0.65).clamp(0.0, 4.0)

        # Flanking diffraction check (+/- 9 canonical cells perpendicular offset) in ONE batched [2, M, 24] lookup
        inv_len = torch.rsqrt(dx * dx + dy * dy + 1e-4)
        px = -dy * inv_len * (9.0 * inv_scale)
        py = dx * inv_len * (9.0 * inv_scale)
        sin_t = self.sin_t_ray
        base_x = (src_xyz[:, 0:1] + t * dx.unsqueeze(1)).unsqueeze(0)  # [1, M, 24]
        base_y = (src_xyz[:, 1:2] + t * dy.unsqueeze(1)).unsqueeze(0)
        rx_o = (base_x + self.signs_2 * (px.unsqueeze(1) * sin_t).unsqueeze(0)).clamp_(0, W - 1).long()
        ry_o = (base_y + self.signs_2 * (py.unsqueeze(1) * sin_t).unsqueeze(0)).clamp_(0, H - 1).long()
        w_o = solid_wall[rz.unsqueeze(0).expand(2, -1, -1), ry_o, rx_o].float()  # [2, M, 24]
        prev_wo = F.pad(w_o[:, :, :-1], (1, 0), value=0.0)
        nw_o = F.relu(w_o - prev_wo).sum(dim=2).amin(dim=0)  # min walls across left/right flank
        diffract_relief = ((n_walls >= 0.8) & (nw_o < n_walls - 0.5)).float() * 4.5

        pl_wall = torch.clamp(n_walls * WALL_LOSS_DB - diffract_relief, min=0.0)

        # Vertical ceiling / plateau slab / open shaft coupling
        s_ix = src_xyz[:, 0].clamp(0, W - 1).long()
        s_iy = src_xyz[:, 1].clamp(0, H - 1).long()
        d_ix = dst_xyz[:, 0].clamp(0, W - 1).long()
        d_iy = dst_xyz[:, 1].clamp(0, H - 1).long()
        s_iz = src_xyz[:, 2].clamp(0, L - 1).long()
        d_iz = dst_xyz[:, 2].clamp(0, L - 1).long()

        lo = torch.minimum(s_iz, d_iz)
        hi = torch.maximum(s_iz, d_iz)

        cross_01 = (lo == 0) & (hi >= 1)
        shaft_ok = torch.maximum(open_shaft_01[s_iy, s_ix].float(), open_shaft_01[d_iy, d_ix].float()) > 0.5
        pl_01 = cross_01.float() * (CAVE_CEIL_LOSS_DB - (CAVE_CEIL_LOSS_DB - OPEN_VERT_LOSS_DB) * shaft_ok.float())
        ceiling_muffled = (cross_01 & (~shaft_ok)).float()

        cross_12 = (lo <= 1) & (hi >= 2)
        sky_ok = torch.maximum(open_sky_12[s_iy, s_ix].float(), open_sky_12[d_iy, d_ix].float()) > 0.5
        pl_12 = cross_12.float() * (PLATEAU_SLAB_LOSS_DB - (PLATEAU_SLAB_LOSS_DB - OPEN_VERT_LOSS_DB) * sky_ok.float())

        cross_23 = (lo <= 2) & (hi >= 3)
        pl_vert = pl_01 + pl_12 + cross_23.float() * OPEN_VERT_LOSS_DB

        atten_db = pl_geom + pl_wall + pl_vert
        return {
            "dist": dist,
            "atten_db": atten_db,
            "walls_crossed": n_walls,
            "ceiling_muffled": ceiling_muffled,
            "dst_noise_db": noise_db[d_iz, d_iy, d_ix].float(),
            "dst_noise_cause": noise_cause[d_iz, d_iy, d_ix].long(),
            "src_noise_db": noise_db[s_iz, s_iy, s_ix].float(),
        }


# =============================================================================
# 3. Neural Models: CommBotPair, ViewerInterpreter, HumanCommandEncoder
# =============================================================================
class EgoWorldEncoder(nn.Module):
    """Lightweight Conv encoder for WorldModel Z_world [N, 48, 16, 16] ego-crops."""

    def __init__(self, in_ch: int = 48, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(out_dim * 2, out_dim)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        h = self.net(crops.float())
        return self.proj(torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=-1))


class CommBotPair(nn.Module):
    """
    Emergent discrete-symbol (K=16) + 8-d prosody + adaptive volume Spotter/Teammate
    pair trained through the differentiable physical AcousticField3D SNR channel.
    """

    def __init__(self, num_symbols: int = 16, prosody_dim: int = 8):
        super().__init__()
        self.num_symbols = num_symbols
        self.prosody_dim = prosody_dim
        self.utterance_dim = num_symbols + prosody_dim

        self.spot_world_enc = EgoWorldEncoder(48, 64)
        self.spot_mlp = nn.Sequential(
            nn.Linear(64 + 10, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
        )
        self.sym_head = nn.Linear(96, num_symbols)
        self.prosody_head = nn.Linear(96, prosody_dim)
        self.vol_head = nn.Sequential(
            nn.Linear(96 + 6, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )
        nn.init.constant_(self.vol_head[-1].bias, 1.4)

        self.team_world_enc = EgoWorldEncoder(48, 64)
        self.team_mlp = nn.Sequential(
            nn.Linear(64 + 4 + self.utterance_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 96),
            nn.GELU(),
        )
        self.sector_head = nn.Linear(96, len(SECTOR_NAMES))
        self.elem_head = nn.Linear(96, len(ELEMENT_NAMES))
        self.tactic_head = nn.Linear(96, len(TACTIC_NAMES))

    def encode_spotter(
        self,
        spot_crops: torch.Tensor,
        spot_aux: torch.Tensor,
        tau: float = 1.0,
        hard_eval: bool = False,
    ) -> dict[str, torch.Tensor]:
        hs_w = self.spot_world_enc(spot_crops)
        hs = self.spot_mlp(torch.cat([hs_w, spot_aux], dim=-1))
        sym_logits = self.sym_head(hs)
        if hard_eval or not self.training:
            idx = sym_logits.argmax(dim=-1)
            sym_onehot = F.one_hot(idx, num_classes=self.num_symbols).float()
        else:
            sym_onehot = F.gumbel_softmax(sym_logits, tau=tau, hard=True, dim=-1)

        prosody_raw = torch.tanh(self.prosody_head(hs))
        prosody = prosody_raw / (prosody_raw.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(0.95)
        vol = torch.sigmoid(self.vol_head(torch.cat([hs, spot_aux[:, 4:10]], dim=-1))).squeeze(-1)
        u_tx = torch.cat([sym_onehot * 1.25, prosody], dim=-1)
        return {
            "sym_logits": sym_logits,
            "sym_id": sym_logits.argmax(dim=-1),
            "prosody": prosody,
            "volume": vol,
            "u_tx": u_tx,
        }

    def acoustic_channel(
        self,
        u_tx: torch.Tensor,
        volume: torch.Tensor,
        atten_tm: torch.Tensor,
        noise_tm: torch.Tensor,
        atten_en: torch.Tensor | None = None,
        noise_en: torch.Tensor | None = None,
        deterministic_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        src_spl = SRC_SPL_MIN_DB + SRC_SPL_SPAN_DB * volume
        snr_tm = src_spl - atten_tm - noise_tm
        snr_clamped = torch.clamp(snr_tm, -25.0, 35.0)
        snr_lin = torch.pow(10.0, snr_clamped / 20.0)
        audibility = torch.sigmoid((snr_tm + 2.0) / 3.5).unsqueeze(-1)
        noise_std = torch.clamp(1.0 / (snr_lin + 0.14), 0.03, 3.8).unsqueeze(-1)
        if deterministic_noise:
            gen = torch.Generator(device=u_tx.device).manual_seed(1337)
            eps = torch.randn(u_tx.shape, device=u_tx.device, dtype=u_tx.dtype, generator=gen)
        else:
            eps = torch.randn_like(u_tx)
        u_rx = audibility * u_tx + noise_std * eps
        u_heard = u_rx / torch.sqrt(1.0 + noise_std * noise_std)

        out = {"u_heard": u_heard, "snr_tm": snr_tm, "volume_used": volume}
        if atten_en is not None and noise_en is not None:
            snr_en = src_spl - atten_en - noise_en
            out["snr_en"] = snr_en
            out["eavesdrop_prob"] = torch.sigmoid((snr_en - ENEMY_DETECT_SNR_DB) / 2.5)
        return out

    def decode_teammate(
        self,
        team_crops: torch.Tensor,
        team_aux: torch.Tensor,
        u_heard: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ht_w = self.team_world_enc(team_crops)
        ht = self.team_mlp(torch.cat([ht_w, team_aux, u_heard], dim=-1))
        return self.sector_head(ht), self.elem_head(ht), self.tactic_head(ht)

    def forward(
        self,
        spot_crops: torch.Tensor,
        spot_aux: torch.Tensor,
        team_crops: torch.Tensor,
        team_aux: torch.Tensor,
        atten_tm: torch.Tensor,
        noise_tm: torch.Tensor,
        atten_en: torch.Tensor,
        noise_en: torch.Tensor,
        tau: float = 1.0,
        deterministic_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        sp = self.encode_spotter(spot_crops, spot_aux, tau=tau)
        ch = self.acoustic_channel(
            sp["u_tx"], sp["volume"], atten_tm, noise_tm, atten_en, noise_en, deterministic_noise=deterministic_noise
        )
        sec_l, el_l, tac_l = self.decode_teammate(team_crops, team_aux, ch["u_heard"])
        return {**sp, **ch, "sector_logits": sec_l, "elem_logits": el_l, "tactic_logits": tac_l}


class ViewerInterpreter(nn.Module):
    """Lightweight viewer-side neural interpreter: (u_heard, SNR, speaker_cls, z_ctx) -> semantic frame."""

    def __init__(self, utterance_dim: int = 24, n_classes: int = 4, z_dim: int = 48):
        super().__init__()
        in_dim = utterance_dim + 1 + n_classes + z_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 96),
            nn.GELU(),
        )
        self.event_head = nn.Linear(96, len(EVENT_NAMES))
        self.sector_head = nn.Linear(96, len(SECTOR_NAMES))
        self.elem_head = nn.Linear(96, len(ELEMENT_NAMES))
        self.tactic_head = nn.Linear(96, len(TACTIC_NAMES))
        self.urgency_head = nn.Linear(96, 1)
        self.tier_head = nn.Linear(96, 3)

    def forward(
        self,
        u_heard: torch.Tensor,
        snr_db: torch.Tensor,
        speaker_cls: torch.Tensor,
        z_ctx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cls_oh = F.one_hot(speaker_cls.clamp(0, len(SPEAKER_CLASSES) - 1), num_classes=len(SPEAKER_CLASSES)).float()
        snr_f = torch.clamp(snr_db / 25.0, -1.5, 1.5).unsqueeze(-1)
        h = self.trunk(torch.cat([u_heard, snr_f, cls_oh, z_ctx], dim=-1))
        return {
            "event_logits": self.event_head(h),
            "sector_logits": self.sector_head(h),
            "elem_logits": self.elem_head(h),
            "tactic_logits": self.tactic_head(h),
            "urgency": torch.sigmoid(self.urgency_head(h)).squeeze(-1),
            "tier_logits": self.tier_head(h),
        }


class HumanCommandEncoder(nn.Module):
    """
    Fast neural Human Chat -> 16-d Goal Vector `g_i` + Direct Tactical Override Decoder.
    Outputs:
      - g_i          : [B, 16] continuous goal vector in [-1, 1]
      - target_xy    : [B, 2] normalized map target in [0, 1]^2
      - elem_logits  : [B, 3] (Fire, Frost, Oil)
      - ability_logits: [B, 6] (dash, melee, cast_bolt, carve_ice, ability_e, ult_r)
      - intent_logits: [B, num_intents]
    """

    def __init__(self, bow_dim: int = 256, goal_dim: int = 16, num_intents: int = len(LIVE_HUMAN_INTENTS)):
        super().__init__()
        self.bow_dim = bow_dim
        self.goal_dim = goal_dim
        self.num_intents = num_intents
        self.encoder = nn.Sequential(
            nn.Linear(bow_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, goal_dim),
            nn.Tanh(),
        )
        self.dec = nn.Sequential(
            nn.Linear(goal_dim, 48),
            nn.GELU(),
        )
        self.target_head = nn.Linear(48, 2)
        self.elem_head = nn.Linear(48, len(ELEMENT_NAMES))
        self.ability_head = nn.Linear(48, len(ABILITY_NAMES))
        self.intent_head = nn.Linear(48, num_intents)

    def forward(self, text_bow: torch.Tensor) -> dict[str, torch.Tensor]:
        g_vec = self.encoder(text_bow)
        h = self.dec(g_vec)
        return {
            "g_vec": g_vec,
            "target_xy": torch.sigmoid(self.target_head(h)),
            "elem_logits": self.elem_head(h),
            "ability_logits": self.ability_head(h),
            "intent_logits": self.intent_head(h),
        }


# =============================================================================
# 4. Grounded Speech Renderer & Helper Utilities
# =============================================================================
def _snr_tier_label(snr_db: float) -> str:
    if snr_db >= 10.0:
        return "clear"
    if snr_db >= 2.0:
        return "partial"
    if snr_db >= -6.0:
        return "muffled"
    return "inaudible"


def _build_cause_text(cause_id: int, walls_crossed: float, ceiling_muffled: bool) -> str:
    cause_str = CAUSE_DESCRIPTIONS.get(int(cause_id), "ambient interference")
    reasons: list[str] = []
    if ceiling_muffled:
        reasons.append("L0 rock ceiling")
    if walls_crossed >= 0.8:
        w_cnt = max(1, int(round(walls_crossed)))
        reasons.append(f"{w_cnt} basalt wall{'s' if w_cnt > 1 else ''}")
    if int(cause_id) in (1, 2, 3):
        reasons.append(cause_str)
    if not reasons:
        reasons.append(cause_str)
    return " + ".join(reasons)


def render_speech_bubble(
    pred_event: int,
    pred_sector: int,
    pred_elem: int,
    snr_db: float,
    cause_id: int,
    walls_crossed: float,
    ceiling_muffled: bool,
    speaker_name: str,
    src_layer: int,
    volume: float,
) -> tuple[str, str, str]:
    """
    Render grounded English speech bubble with exact 4-tier physical SNR degradation:
      >= +10 dB : crisp tactical callout
      +2..+10 dB: partial interference
      -6..+2 dB : muffled fragments
      < -6 dB   : [inaudible — drowned out by <cause>]
    Returns (bubble_text, tier, cause_text).
    """
    reason_txt = _build_cause_text(cause_id, walls_crossed, ceiling_muffled)
    tier = _snr_tier_label(snr_db)
    tone = "SHOUT" if volume >= 0.72 else ("CALL" if volume >= 0.38 else "whisper")
    sec = SECTOR_NAMES[int(pred_sector) % len(SECTOR_NAMES)]
    elem = ELEMENT_NAMES[int(pred_elem) % len(ELEMENT_NAMES)]
    ev = int(pred_event) % len(EVENT_NAMES)

    if snr_db < -6.0:
        return (
            f"[{speaker_name} @ L{src_layer} ({tone}) | {snr_db:+.1f} dB] [inaudible — drowned out by {reason_txt}]",
            tier,
            reason_txt,
        )

    high_templates = {
        0: f"Oil slick pooled across {sec} — ignite it with {elem} now!",
        1: f"Wildfire pushing through {sec} — suppress the flank with {elem}!",
        2: f"Frozen river bridge spotted near {sec} — shatter the ice with {elem}!",
        3: f"Enemy ambush staged at {sec} shrine — coat the choke in {elem}!",
        4: f"High-ground waterfall choke open at {sec} — lock it down with {elem}!",
        5: f"Cavern vertical shaft clear under {sec} — drop in with {elem} on my mark!",
    }
    med_templates = {
        0: f"Oil slick... {sec}... light it up with {elem}!",
        1: f"Wildfire on the flank... {sec}... hit it with {elem}!",
        2: f"Ice bridge... {sec}... break it with {elem}!",
        3: f"Ambush at {sec}... watch the shrine — deploy {elem}!",
        4: f"Waterfall choke... {sec}... freeze the rim with {elem}!",
        5: f"Cavern shaft below {sec}... drop {elem}!",
    }
    low_fragments = {
        0: f"...oil... [static] ...{sec}... ...{elem}...",
        1: f"...fire... [crackle] ...flank {sec}...",
        2: f"...ice bridge... [muffled] ...{elem}...",
        3: f"...ambush... [static] ...{sec}...",
        4: f"...waterfall... [roar] ...choke {sec}...",
        5: f"...shaft... [echo] ...{sec}...",
    }

    if snr_db >= 10.0:
        return f"[{speaker_name} @ L{src_layer} ({tone}) | {snr_db:+.1f} dB] \"{high_templates[ev]}\"", tier, reason_txt
    if snr_db >= 2.0:
        return (
            f"[{speaker_name} @ L{src_layer} ({tone}) | {snr_db:+.1f} dB, partial {reason_txt}] \"{med_templates[ev]}\"",
            tier,
            reason_txt,
        )
    return (
        f"[{speaker_name} @ L{src_layer} ({tone}) | {snr_db:+.1f} dB, muffled by {reason_txt}] \"{low_fragments[ev]}\"",
        tier,
        reason_txt,
    )


def _cls_to_idx(cls_val: Any) -> int:
    if isinstance(cls_val, (int, np.integer)):
        return int(cls_val) % len(SPEAKER_CLASSES)
    s = str(cls_val).lower()
    if "pyro" in s or "fire" in s:
        return 1
    if "cryo" in s or "frost" in s or "scout" in s:
        return 2
    if "shadow" in s or "flank" in s or "oil" in s or "rogue" in s:
        return 3
    return 0


def _to_grid_xy(x: float, y: float, w: int, h: int, force_norm: bool | None = None) -> tuple[float, float]:
    is_norm = force_norm if force_norm is not None else (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
    if is_norm:
        return float(np.clip(x * (w - 1), 0.0, w - 1)), float(np.clip(y * (h - 1), 0.0, h - 1))
    return float(np.clip(x, 0.0, w - 1)), float(np.clip(y, 0.0, h - 1))


# =============================================================================
# 5. Fast Training Routine (~15-25s on NVIDIA GB10) -> `acoustic_comm_live.pt`
# =============================================================================
def train_and_save(
    ckpt_path: Path | str = DEFAULT_CKPT_PATH,
    world_pt: Path | str = DEFAULT_WORLD_PT,
    device: torch.device | str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Fast end-to-end training of CommBotPair (Spotter/Teammate), ViewerInterpreter,
    and HumanCommandEncoder over real Frostmarsh 3D WorldModel latents + AcousticField3D.
    Trains and saves `acoustic_comm_live.pt` in ~15-25 seconds on NVIDIA GB10.
    """
    t0 = time.perf_counter()
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    from acoustic_comm_poc import (
        build_frostmarsh_3d_state,
        extract_world_latent,
        pos_to_sector,
        sample_z_world_crops,
    )

    wm = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(dev).eval()
    w_ckpt = torch.load(str(world_pt), map_location=dev, weights_only=False)
    wm.load_state_dict(w_ckpt["state_dict"] if isinstance(w_ckpt, dict) and "state_dict" in w_ckpt else w_ckpt)

    acoustics = AcousticField3D(layers=4, n_coarse_steps=8, n_fine_steps=2).to(dev).eval()

    # Build 12 train + 4 validation real Frostmarsh 256x256x4 worlds
    def _build_episodes(seeds: list[int], eps_per_world: int, ep_seed: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(ep_seed)
        sc_l, tc_l, sa_l, ta_l = [], [], [], []
        ev_l, sec_l, el_l, tac_l = [], [], [], []
        atm_l, ntm_l, aen_l, nen_l, cls_l = [], [], [], [], []

        for s in seeds:
            w_state, _, lm = build_frostmarsh_3d_state(s, dev)
            z_world = extract_world_latent(wm, w_state)
            sw, sh01, sk12 = acoustics.extract_acoustic_geometry(w_state, open_shaft_override=lm["open_shaft_01"])
            ndb, ncause = acoustics.compute_ambient_noise_field(w_state, sw, sh01, sk12, use_fp16=(dev.type == "cuda"))

            M = eps_per_world
            spot_xyz = np.zeros((M, 3), dtype=np.float32)
            team_xyz = np.zeros((M, 3), dtype=np.float32)
            en_xyz = np.zeros((M, 3), dtype=np.float32)
            ev_arr = np.zeros(M, dtype=np.int64)
            sec_arr = np.zeros(M, dtype=np.int64)
            amb_arr = np.zeros(M, dtype=np.float32)
            sp_cls = rng.integers(0, len(SPEAKER_CLASSES), size=M, dtype=np.int64)

            for i in range(M):
                ev = i % len(EVENT_NAMES)
                ev_arr[i] = ev
                if ev == 0:
                    ex, ey, ez = lm["oil_positions"][int(rng.integers(0, len(lm["oil_positions"])))]
                elif ev == 1:
                    ex, ey, ez = lm["wildfire_positions"][int(rng.integers(0, len(lm["wildfire_positions"])))]
                elif ev == 2:
                    ex, ey, ez = lm["ice_bridge_pos"]
                elif ev == 3:
                    ex, ey, ez = lm["shrine_pos"]
                    amb_arr[i] = 1.0
                elif ev == 4:
                    ex, ey, ez = lm["waterfall_positions"][int(rng.integers(0, len(lm["waterfall_positions"])))]
                else:
                    ex, ey, ez = lm["shaft_positions"][int(rng.integers(0, len(lm["shaft_positions"])))]

                sx = float(np.clip(ex + rng.uniform(-7.0, 7.0), 12.0, 243.0))
                sy = float(np.clip(ey + rng.uniform(-7.0, 7.0), 12.0, 243.0))
                spot_xyz[i] = (sx, sy, float(ez))
                sec_arr[i] = pos_to_sector(ex, ey)

                ang = rng.uniform(0, 6.28)
                rad = rng.uniform(34.0, 88.0)
                tx = float(np.clip(sx + math.cos(ang) * rad, 14.0, 241.0))
                ty = float(np.clip(sy + math.sin(ang) * rad, 14.0, 241.0))
                tz = int(rng.choice([0, 1, 1, 1, 2]))
                team_xyz[i] = (tx, ty, float(tz))

                e_ang = rng.uniform(0, 6.28)
                e_rad = rng.uniform(28.0, 78.0)
                en_xyz[i] = (
                    float(np.clip(sx + math.cos(e_ang) * e_rad, 10.0, 245.0)),
                    float(np.clip(sy + math.sin(e_ang) * e_rad, 10.0, 245.0)),
                    float(1 if rng.random() < 0.75 else ez),
                )

            s_t = torch.from_numpy(spot_xyz).to(dev)
            t_t = torch.from_numpy(team_xyz).to(dev)
            e_t = torch.from_numpy(en_xyz).to(dev)
            ltm = acoustics.query_pairwise_link(s_t, t_t, sw, sh01, sk12, ndb, ncause)
            len_ = acoustics.query_pairwise_link(s_t, e_t, sw, sh01, sk12, ndb, ncause)

            sc = sample_z_world_crops(z_world, s_t, patch_size=16, radius_cells=16.0).half()
            tc = sample_z_world_crops(z_world, t_t, patch_size=16, radius_cells=16.0).half()

            s_aux = torch.stack(
                [
                    torch.from_numpy(amb_arr).to(dev),
                    s_t[:, 0] / 256.0,
                    s_t[:, 1] / 256.0,
                    s_t[:, 2] / 3.0,
                    (ltm["src_noise_db"] - 50.0) / 30.0,
                    (ltm["atten_db"] - 30.0) / 20.0,
                    (ltm["dst_noise_db"] - 50.0) / 30.0,
                    (len_["atten_db"] - 30.0) / 20.0,
                    (len_["dst_noise_db"] - 50.0) / 30.0,
                    len_["walls_crossed"] / 3.0,
                ],
                dim=-1,
            )
            t_aux = torch.stack(
                [
                    torch.zeros_like(t_t[:, 0]),
                    torch.zeros_like(t_t[:, 1]),
                    t_t[:, 2] / 3.0,
                    (ltm["dst_noise_db"] - 50.0) / 30.0,
                ],
                dim=-1,
            )

            sc_l.append(sc)
            tc_l.append(tc)
            sa_l.append(s_aux)
            ta_l.append(t_aux)
            ev_l.append(torch.from_numpy(ev_arr).to(dev))
            sec_l.append(torch.from_numpy(sec_arr).to(dev))
            el_l.append(torch.tensor([EVENT_TO_ELEM[e] for e in ev_arr], dtype=torch.long, device=dev))
            tac_l.append(torch.tensor([EVENT_TO_TACTIC[e] for e in ev_arr], dtype=torch.long, device=dev))
            atm_l.append(ltm["atten_db"])
            ntm_l.append(ltm["dst_noise_db"])
            aen_l.append(len_["atten_db"])
            nen_l.append(len_["dst_noise_db"])
            cls_l.append(torch.from_numpy(sp_cls).to(dev))

        return {
            "spot_crops": torch.cat(sc_l),
            "team_crops": torch.cat(tc_l),
            "spot_aux": torch.cat(sa_l),
            "team_aux": torch.cat(ta_l),
            "event": torch.cat(ev_l),
            "sector": torch.cat(sec_l),
            "elem": torch.cat(el_l),
            "tactic": torch.cat(tac_l),
            "atten_tm": torch.cat(atm_l),
            "noise_tm": torch.cat(ntm_l),
            "atten_en": torch.cat(aen_l),
            "noise_en": torch.cat(nen_l),
            "speaker_cls": torch.cat(cls_l),
        }

    train_data = _build_episodes(list(range(101, 113)), eps_per_world=600, ep_seed=42)
    val_data = _build_episodes(list(range(901, 905)), eps_per_world=300, ep_seed=99)

    # 1. Train CommBotPair (Spotter + Teammate)
    comm = CommBotPair(num_symbols=16, prosody_dim=8).to(dev)
    opt_c = torch.optim.AdamW(comm.parameters(), lr=2.8e-3, weight_decay=1e-4)
    epochs_c = 24
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=epochs_c)
    N = train_data["event"].shape[0]
    bs = 256
    for ep in range(epochs_c):
        comm.train()
        perm = torch.randperm(N, device=dev)
        tau = max(0.50, 1.8 - 1.3 * (ep / max(1, epochs_c - 1)))
        bal_w = 0.16 * max(0.0, 1.0 - ep / (epochs_c * 0.72))
        st_frac = max(0.0, min(1.0, (ep - 8) / max(1, epochs_c - 11)))
        cur_stealth = 0.28 * st_frac
        for s in range(0, N, bs):
            idx = perm[s : s + bs]
            out = comm(
                train_data["spot_crops"][idx],
                train_data["spot_aux"][idx],
                train_data["team_crops"][idx],
                train_data["team_aux"][idx],
                train_data["atten_tm"][idx],
                train_data["noise_tm"][idx],
                train_data["atten_en"][idx],
                train_data["noise_en"][idx],
                tau=tau,
            )
            loss = (
                F.cross_entropy(out["sector_logits"], train_data["sector"][idx])
                + 0.85 * F.cross_entropy(out["elem_logits"], train_data["elem"][idx])
                + 1.15 * F.cross_entropy(out["tactic_logits"], train_data["tactic"][idx])
            )
            if cur_stealth > 0.0:
                eav_surplus = F.softplus((out["snr_en"] - ENEMY_DETECT_SNR_DB) / 5.0)
                loss = loss + cur_stealth * (0.45 * eav_surplus.mean() + out["eavesdrop_prob"].mean()) + 0.03 * (out["volume"] ** 2).mean()
            else:
                loss = loss + 0.08 * ((out["volume"] - 0.82) ** 2).mean()
            if bal_w > 0:
                probs = F.softmax(out["sym_logits"], dim=-1)
                p_mean = probs.mean(dim=0)
                kl_unif = (p_mean * torch.log(p_mean * comm.num_symbols + 1e-8)).sum()
                ent_s = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                loss = loss + bal_w * kl_unif + 0.015 * ent_s

            opt_c.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(comm.parameters(), 2.0)
            opt_c.step()
        sched_c.step()

    # 2. Train ViewerInterpreter
    comm.eval()
    interp = ViewerInterpreter().to(dev)
    opt_i = torch.optim.AdamW(interp.parameters(), lr=3.2e-3, weight_decay=1e-4)
    epochs_i = 20
    sched_i = torch.optim.lr_scheduler.CosineAnnealingLR(opt_i, T_max=epochs_i)
    for _ in range(epochs_i):
        interp.train()
        perm = torch.randperm(N, device=dev)
        for s in range(0, N, bs):
            idx = perm[s : s + bs]
            with torch.no_grad():
                cout = comm(
                    train_data["spot_crops"][idx],
                    train_data["spot_aux"][idx],
                    train_data["team_crops"][idx],
                    train_data["team_aux"][idx],
                    train_data["atten_tm"][idx],
                    train_data["noise_tm"][idx],
                    train_data["atten_en"][idx],
                    train_data["noise_en"][idx],
                )
            z_ctx = train_data["team_crops"][idx].float().mean(dim=(2, 3))
            pout = interp(cout["u_heard"], cout["snr_tm"], train_data["speaker_cls"][idx], z_ctx)
            gt_urg = torch.clamp(0.35 + 0.55 * cout["volume_used"] + 0.15 * (train_data["event"][idx] == 3).float(), 0.0, 1.0)
            snr_t = cout["snr_tm"]
            gt_tier = torch.zeros_like(snr_t, dtype=torch.long)
            gt_tier = torch.where((snr_t < 5.0) & (snr_t >= -6.0), torch.ones_like(gt_tier), gt_tier)
            gt_tier = torch.where(snr_t < -6.0, torch.full_like(gt_tier, 2), gt_tier)

            loss = (
                F.cross_entropy(pout["event_logits"], train_data["event"][idx])
                + F.cross_entropy(pout["sector_logits"], train_data["sector"][idx])
                + 0.7 * F.cross_entropy(pout["elem_logits"], train_data["elem"][idx])
                + 0.9 * F.cross_entropy(pout["tactic_logits"], train_data["tactic"][idx])
                + 0.5 * F.mse_loss(pout["urgency"], gt_urg)
                + 0.6 * F.cross_entropy(pout["tier_logits"], gt_tier)
            )
            opt_i.zero_grad(set_to_none=True)
            loss.backward()
            opt_i.step()
        sched_i.step()

    # 3. Train HumanCommandEncoder (with word-drop augmentation for robust natural-language generalization)
    human_enc = HumanCommandEncoder(bow_dim=256, goal_dim=16, num_intents=len(LIVE_HUMAN_INTENTS)).to(dev)
    opt_h = torch.optim.AdamW(human_enc.parameters(), lr=4.0e-3, weight_decay=1e-4)

    aug_texts: list[str] = []
    aug_intents: list[int] = []
    rng_txt = np.random.default_rng(seed + 7)
    for k, cmds in enumerate(LIVE_COMMAND_BANK):
        for c in cmds:
            aug_texts.append(c)
            aug_intents.append(k)
            words = c.split()
            if len(words) >= 4:
                for _ in range(2):
                    keep = [w for w in words if rng_txt.random() > 0.22]
                    if len(keep) >= 2:
                        aug_texts.append(" ".join(keep))
                        aug_intents.append(k)

    bow_all = encode_text_hash_bow(aug_texts, dim=256, device=dev)
    int_all = torch.tensor(aug_intents, dtype=torch.long, device=dev)
    tgt_xy_tbl = torch.tensor([[sp[1], sp[2]] for sp in LIVE_HUMAN_INTENTS], dtype=torch.float32, device=dev)
    elem_tbl = torch.tensor([sp[3] for sp in LIVE_HUMAN_INTENTS], dtype=torch.long, device=dev)
    ab_tbl = torch.zeros((len(LIVE_HUMAN_INTENTS), len(ABILITY_NAMES)), dtype=torch.float32, device=dev)
    for k, sp in enumerate(LIVE_HUMAN_INTENTS):
        ab_tbl[k, sp[4]] = 1.0

    for _ in range(260):
        human_enc.train()
        idx = torch.randint(0, len(aug_texts), (128,), device=dev)
        k_gt = int_all[idx]
        hout = human_enc(bow_all[idx])
        loss = (
            1.2 * F.cross_entropy(hout["intent_logits"], k_gt)
            + 1.0 * F.cross_entropy(hout["elem_logits"], elem_tbl[k_gt])
            + 1.0 * F.binary_cross_entropy_with_logits(hout["ability_logits"], ab_tbl[k_gt])
            + 2.5 * F.mse_loss(hout["target_xy"], tgt_xy_tbl[k_gt])
        )
        opt_h.zero_grad(set_to_none=True)
        loss.backward()
        opt_h.step()

    # Evaluate on held-out validation worlds
    comm.eval()
    interp.eval()
    human_enc.eval()
    with torch.no_grad():
        vout = comm(
            val_data["spot_crops"],
            val_data["spot_aux"],
            val_data["team_crops"],
            val_data["team_aux"],
            val_data["atten_tm"],
            val_data["noise_tm"],
            val_data["atten_en"],
            val_data["noise_en"],
            deterministic_noise=True,
        )
        ok_s = vout["sector_logits"].argmax(-1) == val_data["sector"]
        ok_e = vout["elem_logits"].argmax(-1) == val_data["elem"]
        ok_t = vout["tactic_logits"].argmax(-1) == val_data["tactic"]
        joint_acc = float((ok_s & ok_e & ok_t).float().mean().item())
        hi_m = vout["snr_tm"] >= 10.0
        joint_hi_acc = float((ok_s & ok_e & ok_t)[hi_m].float().mean().item()) if hi_m.any() else joint_acc

        z_ctx_v = val_data["team_crops"].float().mean(dim=(2, 3))
        iout = interp(vout["u_heard"], vout["snr_tm"], val_data["speaker_cls"], z_ctx_v)
        interp_ev_acc = float((iout["event_logits"].argmax(-1) == val_data["event"]).float().mean().item())
        interp_ev_hi = (
            float((iout["event_logits"].argmax(-1) == val_data["event"])[hi_m].float().mean().item())
            if hi_m.any()
            else interp_ev_acc
        )

        # Verify key unseen/canonical human commands
        probe_cmds = [
            ("ignite oil at central spire", 0),
            ("freeze north bridge", 1),
            ("retreat to sanctuary", 2),
            ("burn the oil pool at the center spire", 0),
            ("ice over the northern bridge now", 1),
            ("fall back to base sanctuary", 2),
        ]
        p_bow = encode_text_hash_bow([c[0] for c in probe_cmds], dim=256, device=dev)
        p_out = human_enc(p_bow)
        p_acc = float((p_out["intent_logits"].argmax(-1).cpu().numpy() == np.array([c[1] for c in probe_cmds])).mean())

    elapsed = round(time.perf_counter() - t0, 2)
    meta = {
        "version": "live_acoustics_v1",
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "train_elapsed_sec": elapsed,
        "val_joint_coord_acc": round(joint_acc, 4),
        "val_joint_coord_acc_hi_snr": round(joint_hi_acc, 4),
        "val_viewer_event_acc": round(interp_ev_acc, 4),
        "val_viewer_event_acc_hi_snr": round(interp_ev_hi, 4),
        "human_cmd_probe_acc": round(p_acc, 4),
    }

    out_path = Path(ckpt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "comm_pair": comm.state_dict(),
            "viewer_interp": interp.state_dict(),
            "human_encoder": human_enc.state_dict(),
            "meta": meta,
        },
        str(out_path),
    )
    print(
        f"[live_acoustics] Trained & saved {out_path} in {elapsed:.2f}s | "
        f"val_joint={joint_acc*100:.1f}% (hiSNR={joint_hi_acc*100:.1f}%) | "
        f"viewer_ev={interp_ev_acc*100:.1f}% (hiSNR={interp_ev_hi*100:.1f}%) | "
        f"human_probe={p_acc*100:.1f}%"
    )
    return meta


# =============================================================================
# 6. Production Runtime Engine (`LiveAcousticEngine`) — <1.5 ms at 10 Hz!
# =============================================================================
class LiveAcousticEngine:
    """
    Production 10 Hz acoustic speech & human-command steering engine.
    Loads `acoustic_comm_live.pt` (or trains & saves it automatically if missing).
    """

    def __init__(
        self,
        ckpt_path: Path | str = DEFAULT_CKPT_PATH,
        device: torch.device | str | None = None,
        use_fp16: bool = True,
        auto_train: bool = True,
        world_pt: Path | str = DEFAULT_WORLD_PT,
    ) -> None:
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_fp16 = bool(use_fp16) and (self.device.type == "cuda")
        self.ckpt_path = Path(ckpt_path)

        if not self.ckpt_path.exists():
            if not auto_train:
                raise FileNotFoundError(f"Missing acoustic checkpoint: {self.ckpt_path}")
            train_and_save(self.ckpt_path, world_pt=world_pt, device=self.device)

        ckpt = torch.load(str(self.ckpt_path), map_location=self.device, weights_only=False)
        self.meta: dict[str, Any] = ckpt.get("meta", {})

        self.acoustics = AcousticField3D(layers=4, n_coarse_steps=8, n_fine_steps=2).to(self.device).eval()

        self.comm_pair = CommBotPair(num_symbols=16, prosody_dim=8).to(self.device).eval()
        self.comm_pair.load_state_dict(ckpt["comm_pair"])

        self.viewer_interp = ViewerInterpreter().to(self.device).eval()
        self.viewer_interp.load_state_dict(ckpt["viewer_interp"])

        self.human_encoder = HumanCommandEncoder(
            bow_dim=256, goal_dim=16, num_intents=len(LIVE_HUMAN_INTENTS)
        ).to(self.device).eval()
        self.human_encoder.load_state_dict(ckpt["human_encoder"])

        for m in (self.acoustics, self.comm_pair, self.viewer_interp, self.human_encoder):
            for p in m.parameters():
                p.requires_grad_(False)

        # Pre-cached 16x16 crop grid for zero-alloc ego-crop sampling
        lin = torch.linspace(-1.0, 1.0, 16, device=self.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self._base_gx = gx.unsqueeze(0).contiguous()  # [1, 16, 16]
        self._base_gy = gy.unsqueeze(0).contiguous()  # [1, 16, 16]

        # Small cache for encoded human chat BoW + decoder output (instant on repeated texts)
        self._last_chat_text: str | None = None
        self._last_chat_decoded: dict[str, Any] | None = None

        # Pre-captured CUDA Graph for fixed-batch k=2 (Spotter -> AcousticChannel -> ViewerInterpreter)
        self._speech_graph: dict[str, Any] | None = None
        if self.device.type == "cuda":
            self._init_speech_cuda_graph()

    def _init_speech_cuda_graph(self) -> None:
        """Capture a static k=2 CUDA Graph for Spotter -> Channel -> ViewerInterpreter (~0.05 ms replay!)."""
        dev = self.device
        st_crops = torch.zeros((2, 48, 16, 16), device=dev, dtype=torch.float32)
        st_aux = torch.zeros((2, 10), device=dev, dtype=torch.float32)
        st_atten = torch.full((2,), 25.0, device=dev, dtype=torch.float32)
        st_noise = torch.full((2,), 35.0, device=dev, dtype=torch.float32)
        st_cls = torch.zeros((2,), device=dev, dtype=torch.long)
        st_zctx = torch.zeros((2, 48), device=dev, dtype=torch.float32)
        # Fixed deterministic noise vector (reproducible & CUDA-graph safe)
        gen = torch.Generator(device=dev).manual_seed(1337)
        st_eps = torch.randn((2, 24), device=dev, dtype=torch.float32, generator=gen)

        def _body() -> torch.Tensor:
            sp = self.comm_pair.encode_spotter(st_crops, st_aux, hard_eval=True)
            src_spl = SRC_SPL_MIN_DB + SRC_SPL_SPAN_DB * sp["volume"]
            snr_tm = src_spl - st_atten - st_noise
            snr_clamped = torch.clamp(snr_tm, -25.0, 35.0)
            snr_lin = torch.pow(10.0, snr_clamped * 0.05)
            aud = torch.sigmoid((snr_tm + 2.0) / 3.5).unsqueeze(-1)
            n_std = torch.clamp(1.0 / (snr_lin + 0.14), 0.03, 3.8).unsqueeze(-1)
            u_heard = (aud * sp["u_tx"] + n_std * st_eps) * torch.rsqrt(1.0 + n_std * n_std)
            vi = self.viewer_interp(u_heard, snr_tm, st_cls, st_zctx)
            # Pack all 8 scalar outputs into ONE [2, 8] float32 tensor for a single D2H transfer!
            return torch.stack(
                [
                    sp["sym_id"].float(),
                    sp["volume"],
                    snr_tm,
                    vi["event_logits"].argmax(-1).float(),
                    vi["sector_logits"].argmax(-1).float(),
                    vi["elem_logits"].argmax(-1).float(),
                    vi["tactic_logits"].argmax(-1).float(),
                    vi["urgency"],
                ],
                dim=-1,
            )

        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                _ = _body()
        torch.cuda.current_stream(dev).wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            st_out = _body()

        self._speech_graph = {
            "graph": g,
            "crops": st_crops,
            "aux": st_aux,
            "atten": st_atten,
            "noise": st_noise,
            "cls": st_cls,
            "zctx": st_zctx,
            "out": st_out,
        }

    def _run_speech_models(
        self,
        spot_crops: torch.Tensor,  # [k, 48, 16, 16] (k in {1, 2})
        spot_aux: torch.Tensor,    # [k, 10]
        atten_lis: torch.Tensor,   # [k]
        noise_lis: torch.Tensor,   # [k]
        sp_cls: torch.Tensor,      # [k]
        z_ctx: torch.Tensor,       # [k, 48]
    ) -> torch.Tensor:
        """Run Spotter -> Physical Acoustic Channel -> ViewerInterpreter returning packed [k, 8] tensor."""
        k = spot_crops.shape[0]
        if self._speech_graph is not None and k <= 2:
            sg = self._speech_graph
            sg["crops"][:k].copy_(spot_crops.float(), non_blocking=True)
            sg["aux"][:k].copy_(spot_aux, non_blocking=True)
            sg["atten"][:k].copy_(atten_lis, non_blocking=True)
            sg["noise"][:k].copy_(noise_lis, non_blocking=True)
            sg["cls"][:k].copy_(sp_cls, non_blocking=True)
            sg["zctx"][:k].copy_(z_ctx, non_blocking=True)
            sg["graph"].replay()
            return sg["out"][:k]

        sp = self.comm_pair.encode_spotter(spot_crops, spot_aux, hard_eval=True)
        ch = self.comm_pair.acoustic_channel(
            sp["u_tx"], sp["volume"], atten_lis, noise_lis, deterministic_noise=True
        )
        vi = self.viewer_interp(ch["u_heard"], ch["snr_tm"], sp_cls, z_ctx)
        return torch.stack(
            [
                sp["sym_id"].float(),
                sp["volume"],
                ch["snr_tm"],
                vi["event_logits"].argmax(-1).float(),
                vi["sector_logits"].argmax(-1).float(),
                vi["elem_logits"].argmax(-1).float(),
                vi["tactic_logits"].argmax(-1).float(),
                vi["urgency"],
            ],
            dim=-1,
        )

    def _sample_z_crops_fast(self, z_world: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """Zero-copy FP16/FP32 5D grid_sample of [N, 48, 16, 16] crops from z_world [L, 48, H, W]."""
        L, _, H, W = z_world.shape
        n = xyz.shape[0]
        cx = (xyz[:, 0:1, None] / max(W - 1, 1)) * 2.0 - 1.0
        cy = (xyz[:, 1:2, None] / max(H - 1, 1)) * 2.0 - 1.0
        cz = (xyz[:, 2:3, None] / max(L - 1, 1)) * 2.0 - 1.0
        # Keep physical FOV equivalent to radius=16 cells on 256x256 across all resolutions!
        r = (16.0 / 255.0) * 2.0
        grid_x = cx + self._base_gx * r
        grid_y = cy + self._base_gy * r
        grid_z = cz.expand(n, 16, 16)

        z_vol = z_world.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 48, L, H, W] zero-copy view!
        grid_3d = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0).to(dtype=z_vol.dtype)
        crops = F.grid_sample(z_vol, grid_3d, mode="bilinear", padding_mode="border", align_corners=True)
        return crops.squeeze(0).permute(1, 0, 2, 3)  # [N, 48, 16, 16]

    def _decode_human_command(self, text: str) -> dict[str, Any]:
        """Encode natural-language human command -> 16-d goal vector g_i + tactical intent."""
        if text == self._last_chat_text and self._last_chat_decoded is not None:
            return self._last_chat_decoded
        bow = encode_text_hash_bow([text], dim=256, device=self.device)
        hout = self.human_encoder(bow)
        intent_id = int(hout["intent_logits"].argmax(dim=-1).item())
        elem_id = int(hout["elem_logits"].argmax(dim=-1).item())
        ab_probs = torch.sigmoid(hout["ability_logits"]).squeeze(0).float().cpu().numpy()
        g_vec = hout["g_vec"].squeeze(0).float().cpu().numpy()
        pred_xy = hout["target_xy"].squeeze(0).float().cpu().numpy()

        spec = LIVE_HUMAN_INTENTS[intent_id]
        decoded = {
            "intent_id": intent_id,
            "intent_name": spec[0],
            "target_x_norm": float(pred_xy[0]),
            "target_y_norm": float(pred_xy[1]),
            "mirror_team1_x": bool(spec[5]),
            "elem_override": elem_id,
            "elem_name": ELEMENT_NAMES[elem_id],
            "ability_boost": [round(float(v), 4) for v in ab_probs],
            "primary_ability": ABILITY_NAMES[int(np.argmax(ab_probs))],
            "g_vec": [round(float(v), 4) for v in g_vec],
            "action_summary": spec[6],
        }
        self._last_chat_text = text
        self._last_chat_decoded = decoded
        return decoded

    @torch.inference_mode()
    def step_live(
        self,
        w_state: torch.Tensor,
        z_world: torch.Tensor,
        bot_states: list[dict[str, Any]],
        listener_pos: tuple[float, float, float],
        human_chat: dict[str, Any] | None = None,
        max_bubbles: int = 2,
    ) -> dict[str, Any]:
        """
        Run one live 10 Hz acoustic tick (<1.5 ms on NVIDIA GB10 for both 256x256x4 & 512x512x4).

        Args:
            w_state      : [L, 18, H, W] float16/float32 on CUDA
            z_world      : [L, 48, H, W] float16/float32 on CUDA
            bot_states   : list of dicts {id, name, team, cls, x, y, layer, hp}
            listener_pos : (x, y, layer) normalized [0..1] or grid units [0..W, 0..H]
            human_chat   : optional {"text": str, "x": float, "y": float, "layer": int, "volume": float, "team": int}

        Returns dict with:
            speech_bubbles     : list of grounded bot speech bubbles (1-2 salient bots)
            human_overrides    : list of per-bot g_i steering overrides (friendly bots with snr_db > 0)
            human_replies      : list of live bot replies to human chat (clear + muffled)
            enemy_eavesdropped : list of enemy bots that overheard the human chat (snr_db > +6 dB)
            ambient_stats      : listener ambient noise dB & dominant noise cause
        """
        L, _, H, W = w_state.shape
        dt = torch.float16 if self.use_fp16 else torch.float32
        w_gpu = w_state.to(device=self.device, dtype=dt, non_blocking=True)
        z_gpu = z_world.to(device=self.device, non_blocking=True)

        # 1. Physical 3D acoustic geometry + 10-step FP16 wave/occlusion ambient noise field (CUDA-graph accelerated!)
        solid_wall, open_shaft_01, open_sky_12, noise_db, noise_cause = self.acoustics.compute_geometry_and_noise(
            w_gpu, use_fp16=self.use_fp16
        )

        # Resolve listener coordinates (auto-detect normalized vs grid units)
        lx, ly = _to_grid_xy(float(listener_pos[0]), float(listener_pos[1]), W, H)
        lz = float(np.clip(listener_pos[2], 0, L - 1))
        Ha, Wa = int(noise_db.shape[1]), int(noise_db.shape[2])
        li_x = int(np.clip(round(lx * (Wa - 1) / max(W - 1, 1)), 0, Wa - 1))
        li_y = int(np.clip(round(ly * (Ha - 1) / max(H - 1, 1)), 0, Ha - 1))
        li_z = int(round(lz))

        alive_bots = [b for b in bot_states if float(b.get("hp", 1.0)) > 0.0]
        if not alive_bots:
            l_ndb = float(noise_db[li_z, li_y, li_x].item())
            l_cause = int(noise_cause[li_z, li_y, li_x].item())
            return {
                "grid": [L, H, W],
                "speech_bubbles": [],
                "human_overrides": [],
                "human_replies": [],
                "enemy_eavesdropped": [],
                "ambient_stats": {
                    "listener_noise_db": round(l_ndb, 2),
                    "listener_noise_cause": CAUSE_DESCRIPTIONS.get(l_cause, "ambient"),
                },
            }

        # Auto-detect normalized [0..1] vs grid coordinates for bot_states collectively
        bots_are_norm = all(
            0.0 <= float(b["x"]) <= 1.0 and 0.0 <= float(b["y"]) <= 1.0 for b in alive_bots
        )
        n_bots = len(alive_bots)
        bot_xyz_np = np.empty((n_bots, 3), dtype=np.float32)
        bot_teams_np = np.empty(n_bots, dtype=np.int64)
        bot_cls_np = np.empty(n_bots, dtype=np.int64)

        for i, b in enumerate(alive_bots):
            bx, by = _to_grid_xy(float(b["x"]), float(b["y"]), W, H, force_norm=bots_are_norm)
            bz = float(np.clip(b.get("layer", 1), 0, L - 1))
            bot_xyz_np[i, 0] = bx
            bot_xyz_np[i, 1] = by
            bot_xyz_np[i, 2] = bz
            bot_teams_np[i] = int(b.get("team", 0))
            bot_cls_np[i] = _cls_to_idx(b.get("cls", 0))

        # Pack bot coords, listener, optional human source, teams/cls, and CPU-computed combat salience into ONE single H2D upload!
        has_chat = human_chat is not None and bool(str(human_chat.get("text", "")).strip())
        if has_chat:
            assert human_chat is not None
            h_text = str(human_chat["text"]).strip()
            hx, hy = _to_grid_xy(float(human_chat.get("x", lx)), float(human_chat.get("y", ly)), W, H)
            hz = float(np.clip(human_chat.get("layer", lz), 0, L - 1))
            h_vol = float(np.clip(human_chat.get("volume", 0.85), 0.0, 1.0))
            h_team = int(human_chat.get("team", 0))
            decoded = self._decode_human_command(h_text)

        # Compute combat proximity & nearest enemy on CPU (N<=64 -> ~3 us, eliminates 8 GPU kernel launches!)
        scale_256 = 256.0 / float(max(H, W))
        dx_m = (bot_xyz_np[:, 0:1] - bot_xyz_np[None, :, 0]) * scale_256
        dy_m = (bot_xyz_np[:, 1:2] - bot_xyz_np[None, :, 1]) * scale_256
        dmat_np = np.hypot(dx_m, dy_m)
        opp_mask_np = bot_teams_np[:, None] != bot_teams_np[None, :]
        nearest_opp_np = np.zeros(n_bots, dtype=np.float32)
        combat_sal_np = np.zeros(n_bots, dtype=np.float32)
        for i in range(n_bots):
            if opp_mask_np[i].any():
                d_row = np.where(opp_mask_np[i], dmat_np[i], 9999.0)
                j_min = int(np.argmin(d_row))
                nearest_opp_np[i] = float(j_min)
                combat_sal_np[i] = math.exp(-float(d_row[j_min]) / 35.0)
            else:
                nearest_opp_np[i] = float(i)

        # Single compact H2D upload [N + 2, 6]: cols 0..3=xyz, 3=cls, 4=team, 5=combat_sal (row N=listener, row N+1=human_src)
        meta_np = np.zeros((n_bots + 2, 7), dtype=np.float32)
        meta_np[:n_bots, :3] = bot_xyz_np
        meta_np[:n_bots, 3] = bot_cls_np
        meta_np[:n_bots, 4] = bot_teams_np
        meta_np[:n_bots, 5] = combat_sal_np
        meta_np[:n_bots, 6] = nearest_opp_np
        meta_np[n_bots, :3] = (lx, ly, lz)
        if has_chat:
            meta_np[n_bots + 1, :3] = (hx, hy, hz)

        meta_t = torch.from_numpy(meta_np).to(self.device, non_blocking=True)
        all_xyz = meta_t[: n_bots + 1, :3]
        bot_xyz = meta_t[:n_bots, :3]
        listener_xyz_1 = meta_t[n_bots : n_bots + 1, :3]
        bot_cls = meta_t[:n_bots, 3].long()
        bot_teams_t = meta_t[:n_bots, 4]
        combat_sal = meta_t[:n_bots, 5]
        nearest_opp_t = meta_t[:n_bots, 6].long()

        # 2. Sample Z_world ego-crops for all bots + listener in ONE single 5D grid_sample call!
        all_crops = self._sample_z_crops_fast(z_gpu, all_xyz)  # [N + 1, 48, 16, 16]
        bot_crops = all_crops[:n_bots]
        lis_ctx_1 = all_crops[n_bots : n_bots + 1].float().mean(dim=(2, 3))  # [1, 48]

        z_sal = bot_crops.float().var(dim=(2, 3)).mean(dim=1)  # [N]

        b_ix = bot_xyz[:, 0].clamp(0, W - 1).long()
        b_iy = bot_xyz[:, 1].clamp(0, H - 1).long()
        b_iz = bot_xyz[:, 2].clamp(0, L - 1).long()

        # Single fancy-index kernel for ALL 18 channels at N bot locations (instead of 5 separate kernels!)
        w_bots = w_gpu[b_iz, :, b_iy, b_ix].float()  # [N, 18]
        phys_sal = (
            1.4 * w_bots[:, S_OIL]
            + 1.2 * torch.clamp(w_bots[:, CH_TEMP] - 1.0, min=0.0)
            + 0.7 * w_bots[:, S_WATER]
            + 0.9 * w_bots[:, S_ICE]
            + 1.5 * w_bots[:, CH_FORCING].abs()
        )

        sal_total = z_sal + phys_sal + 0.8 * combat_sal
        k = min(max(1, max_bubbles), n_bots)
        top1_idx = sal_total.argmax(dim=0, keepdim=True)  # [1]
        if k >= 2 and n_bots >= 2:
            t0_team = bot_teams_t[top1_idx]
            sal2 = sal_total.clone()
            sal2[top1_idx] = -1e9
            sal2 = sal2 + (bot_teams_t != t0_team).float() * 0.35
            top2_idx = sal2.argmax(dim=0, keepdim=True)
            pick_t = torch.cat([top1_idx, top2_idx], dim=0)  # [2]
        else:
            pick_t = top1_idx  # [1]

        spot_xyz = bot_xyz[pick_t]  # [k, 3]
        spot_crops = bot_crops[pick_t]  # [k, 48, 16, 16]
        listener_xyz = listener_xyz_1.expand(k, 3)

        en_idx_k = nearest_opp_t[pick_t]
        en_xyz_t = bot_xyz[en_idx_k]
        ambush_flag = (combat_sal[pick_t] > 0.55).float()

        # 3. Batch ALL acoustic links (speech + enemy + optional human chat) in ONE single CUDA-graphed ray-march!
        if has_chat:
            h_src = meta_t[n_bots + 1 : n_bots + 2, :3].expand(n_bots, 3)
            q_src = torch.cat([spot_xyz, spot_xyz, h_src], dim=0)
            q_dst = torch.cat([listener_xyz, en_xyz_t, bot_xyz], dim=0)
        else:
            q_src = torch.cat([spot_xyz, spot_xyz], dim=0)
            q_dst = torch.cat([listener_xyz, en_xyz_t], dim=0)

        link_all = self.acoustics.query_pairwise_link_fast(
            q_src, q_dst, (H, W), solid_wall, open_shaft_01, open_sky_12, noise_db, noise_cause
        )

        atten_lis = link_all["atten_db"][:k]
        noise_lis = link_all["dst_noise_db"][:k]
        cause_lis = link_all["dst_noise_cause"][:k]
        walls_lis = link_all["walls_crossed"][:k]
        ceil_lis = link_all["ceiling_muffled"][:k]
        src_noise = link_all["src_noise_db"][:k]

        atten_en = link_all["atten_db"][k : 2 * k]
        noise_en = link_all["dst_noise_db"][k : 2 * k]
        walls_en = link_all["walls_crossed"][k : 2 * k]

        spot_aux = torch.stack(
            [
                ambush_flag,
                spot_xyz[:, 0] / float(W),
                spot_xyz[:, 1] / float(H),
                spot_xyz[:, 2] / float(max(L - 1, 1)),
                (src_noise - 50.0) / 30.0,
                (atten_lis - 30.0) / 20.0,
                (noise_lis - 50.0) / 30.0,
                (atten_en - 30.0) / 20.0,
                (noise_en - 50.0) / 30.0,
                walls_en / 3.0,
            ],
            dim=-1,
        )

        # Run Spotter -> Channel -> ViewerInterpreter via CUDA Graph (~0.05 ms!)
        speech_pack = self._run_speech_models(
            spot_crops, spot_aux, atten_lis, noise_lis, bot_cls[pick_t], lis_ctx_1.expand(k, -1)
        )  # [k, 8]

        # Pack ALL readbacks (picked_idx + speech_pack + acoustics + listener ambient + optional human chat SNRs) into ONE D2H copy!
        sp_full = torch.cat(
            [
                pick_t.float().unsqueeze(-1),
                speech_pack,
                cause_lis.float().unsqueeze(-1),
                walls_lis.unsqueeze(-1),
                ceil_lis.unsqueeze(-1),
            ],
            dim=-1,
        )  # [k, 12]
        lis_amb = torch.stack([noise_db[li_z, li_y, li_x].float(), noise_cause[li_z, li_y, li_x].float()])  # [2]

        if has_chat:
            h_spl = SRC_SPL_MIN_DB + SRC_SPL_SPAN_DB * h_vol
            h_snr_t = h_spl - link_all["atten_db"][2 * k :] - link_all["dst_noise_db"][2 * k :]
            h_pack = torch.stack(
                [
                    h_snr_t,
                    link_all["dst_noise_cause"][2 * k :].float(),
                    link_all["walls_crossed"][2 * k :],
                    link_all["ceiling_muffled"][2 * k :],
                ],
                dim=-1,
            )  # [N, 4]
            flat_cpu = torch.cat([sp_full.reshape(-1), lis_amb, h_pack.reshape(-1)]).cpu().numpy()
            sp_np = flat_cpu[: k * 12].reshape(k, 12)
            l_ndb = float(flat_cpu[k * 12])
            l_cause = int(round(float(flat_cpu[k * 12 + 1])))
            h_np = flat_cpu[k * 12 + 2 :].reshape(n_bots, 4)
        else:
            flat_cpu = torch.cat([sp_full.reshape(-1), lis_amb]).cpu().numpy()
            sp_np = flat_cpu[: k * 12].reshape(k, 12)
            l_ndb = float(flat_cpu[k * 12])
            l_cause = int(round(float(flat_cpu[k * 12 + 1])))
            h_np = None

        speech_bubbles: list[dict[str, Any]] = []
        for idx_local in range(k):
            row = sp_np[idx_local]
            b_i = int(round(float(row[0])))
            b = alive_bots[b_i]
            sp_name = str(b.get("name", SPEAKER_CLASSES[int(bot_cls_np[b_i])]))
            src_layer = int(round(float(bot_xyz_np[b_i, 2])))
            sym_id = int(round(float(row[1])))
            vol_v = float(row[2])
            snr_v = float(row[3])
            p_ev = int(round(float(row[4])))
            p_sec = int(round(float(row[5])))
            p_el = int(round(float(row[6])))
            p_tac = int(round(float(row[7])))
            p_urg = float(row[8])
            c_id = int(round(float(row[9])))
            w_cr = float(row[10])
            c_muf = bool(row[11] > 0.5)

            txt, tier, cause_txt = render_speech_bubble(
                pred_event=p_ev,
                pred_sector=p_sec,
                pred_elem=p_el,
                snr_db=snr_v,
                cause_id=c_id,
                walls_crossed=w_cr,
                ceiling_muffled=c_muf,
                speaker_name=sp_name,
                src_layer=src_layer,
                volume=vol_v,
            )
            speech_bubbles.append(
                {
                    "bot_id": b.get("id", b_i),
                    "bot_name": sp_name,
                    "team": int(bot_teams_np[b_i]),
                    "cls": SPEAKER_CLASSES[int(bot_cls_np[b_i])],
                    "pos": [round(float(bot_xyz_np[b_i, 0]), 1), round(float(bot_xyz_np[b_i, 1]), 1), src_layer],
                    "symbol_id": sym_id,
                    "volume": round(vol_v, 3),
                    "snr_db": round(snr_v, 2),
                    "tier": tier,
                    "cause": cause_txt,
                    "event": EVENT_NAMES[p_ev],
                    "sector": SECTOR_NAMES[p_sec],
                    "element": ELEMENT_NAMES[p_el],
                    "tactic": TACTIC_NAMES[p_tac],
                    "urgency": round(p_urg, 3),
                    "walls_crossed": round(w_cr, 2),
                    "ceiling_muffled": c_muf,
                    "text": txt,
                }
            )

        # 4. Assemble Human Chat Overrides & Replies
        human_overrides: list[dict[str, Any]] = []
        human_replies: list[dict[str, Any]] = []
        enemy_eavesdropped: list[dict[str, Any]] = []
        human_intent_meta: dict[str, Any] | None = None

        if has_chat and h_np is not None:
            human_intent_meta = {
                "text": h_text,
                "intent": decoded["intent_name"],
                "elem_override": decoded["elem_override"],
                "elem_name": decoded["elem_name"],
                "primary_ability": decoded["primary_ability"],
                "action_summary": decoded["action_summary"],
                "g_vec": decoded["g_vec"],
                "source_pos": [round(hx, 1), round(hy, 1), int(round(hz))],
                "volume": round(h_vol, 3),
            }

            tgt_x_n = decoded["target_x_norm"]
            tgt_y_n = decoded["target_y_norm"]
            if decoded["mirror_team1_x"] and h_team == 1:
                tgt_x_n = 1.0 - tgt_x_n

            for i, b in enumerate(alive_bots):
                b_team = int(bot_teams_np[i])
                b_name = str(b.get("name", f"Bot-{b.get('id', i)}"))
                b_id = b.get("id", i)
                b_snr = float(h_np[i, 0])
                b_cause = _build_cause_text(int(round(float(h_np[i, 1]))), float(h_np[i, 2]), bool(h_np[i, 3] > 0.5))

                if b_team == h_team:
                    if b_snr > 0.0:
                        bx_n = float(bot_xyz_np[i, 0]) / float(max(W - 1, 1))
                        by_n = float(bot_xyz_np[i, 1]) / float(max(H - 1, 1))
                        mdx = tgt_x_n - bx_n
                        mdy = tgt_y_n - by_n
                        mnorm = math.hypot(mdx, mdy)
                        if mnorm > 1e-3:
                            move_x, move_y = mdx / mnorm, mdy / mnorm
                        else:
                            move_x, move_y = 0.0, 0.0

                        snr_gain = float(1.0 / (1.0 + math.exp(-(b_snr - 2.0) / 3.0)))
                        if b_snr >= 10.0:
                            reply_txt = f"Copy ({b_snr:+.0f}dB)! {decoded['action_summary']}!"
                            r_tier = "clear"
                        else:
                            reply_txt = f"Copy through static ({b_snr:+.0f}dB, {b_cause}) — {decoded['action_summary']}!"
                            r_tier = "partial"

                        human_overrides.append(
                            {
                                "bot_id": b_id,
                                "bot_name": b_name,
                                "team": b_team,
                                "snr_db": round(b_snr, 2),
                                "snr_gain": round(snr_gain, 3),
                                "g_vec": decoded["g_vec"],
                                "move_x": round(move_x, 4),
                                "move_y": round(move_y, 4),
                                "target_xy_norm": [round(tgt_x_n, 3), round(tgt_y_n, 3)],
                                "elem_override": decoded["elem_override"],
                                "elem_name": decoded["elem_name"],
                                "ability_boost": decoded["ability_boost"],
                                "primary_ability": decoded["primary_ability"],
                                "intent": decoded["intent_name"],
                                "reply_text": reply_txt,
                            }
                        )
                        human_replies.append(
                            {
                                "bot_id": b_id,
                                "bot_name": b_name,
                                "team": b_team,
                                "snr_db": round(b_snr, 2),
                                "tier": r_tier,
                                "cause": b_cause,
                                "heard": True,
                                "reply_text": reply_txt,
                            }
                        )
                    elif b_snr >= -6.0:
                        reply_txt = f"[muffled {b_snr:+.0f}dB] Say again?! ({b_cause})"
                        human_replies.append(
                            {
                                "bot_id": b_id,
                                "bot_name": b_name,
                                "team": b_team,
                                "snr_db": round(b_snr, 2),
                                "tier": "muffled",
                                "cause": b_cause,
                                "heard": False,
                                "reply_text": reply_txt,
                            }
                        )
                    else:
                        reply_txt = f"[inaudible {b_snr:+.0f}dB — drowned out by {b_cause}]"
                        human_replies.append(
                            {
                                "bot_id": b_id,
                                "bot_name": b_name,
                                "team": b_team,
                                "snr_db": round(b_snr, 2),
                                "tier": "inaudible",
                                "cause": b_cause,
                                "heard": False,
                                "reply_text": reply_txt,
                            }
                        )
                else:
                    if b_snr > ENEMY_DETECT_SNR_DB:
                        enemy_eavesdropped.append(
                            {
                                "bot_id": b_id,
                                "bot_name": b_name,
                                "team": b_team,
                                "snr_db": round(b_snr, 2),
                                "alert": f"Enemy {b_name} overheard human comms (+{b_snr:.1f} dB)!",
                            }
                        )

        return {
            "grid": [L, H, W],
            "speech_bubbles": speech_bubbles,
            "human_intent": human_intent_meta,
            "human_overrides": human_overrides,
            "human_replies": human_replies,
            "enemy_eavesdropped": enemy_eavesdropped,
            "ambient_stats": {
                "listener_noise_db": round(l_ndb, 2),
                "listener_noise_cause": CAUSE_DESCRIPTIONS.get(l_cause, "ambient"),
            },
        }


# =============================================================================
# 7. End-to-End Verification & Benchmark Suite -> `ITER3_SPARK_VERIFY.json`
# =============================================================================
def run_full_verification(
    out_json: Path | str = ROOT / "ITER3_SPARK_VERIFY.json",
    retrain: bool = False,
) -> dict[str, Any]:
    import json
    from acoustic_comm_poc import build_frostmarsh_3d_state
    from sparse_world_runner import SparseWorldRunner, dense_world_forward_with_latent, verify_sparse_world_runner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print("\n=== [1/3] Verifying SparseWorldRunner (256x256x4 & 512x512x4) ===")
    sparse_verify = verify_sparse_world_runner(DEFAULT_WORLD_PT)

    print("\n=== [2/3] Training / Loading acoustic_comm_live.pt ===")
    if retrain or not DEFAULT_CKPT_PATH.exists():
        train_meta = train_and_save(DEFAULT_CKPT_PATH, DEFAULT_WORLD_PT, device=device)
    else:
        ckpt = torch.load(str(DEFAULT_CKPT_PATH), map_location="cpu", weights_only=False)
        train_meta = ckpt.get("meta", {})
        print(f"[live_acoustics] Loaded existing {DEFAULT_CKPT_PATH}: {train_meta}")

    engine = LiveAcousticEngine(DEFAULT_CKPT_PATH, device=device, use_fp16=True, auto_train=False)

    # Load WorldModel for generating genuine Z_world at both 256x256x4 and 512x512x4
    wm = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device).half().eval()
    w_ckpt = torch.load(str(DEFAULT_WORLD_PT), map_location=device, weights_only=False)
    wm.load_state_dict(w_ckpt["state_dict"] if isinstance(w_ckpt, dict) and "state_dict" in w_ckpt else w_ckpt)

    print("\n=== [3/3] Benchmarking LiveAcousticEngine.step_live() & Human Commands ===")
    w256, _, lm256 = build_frostmarsh_3d_state(912, device)
    w256_h = w256.half().contiguous()
    with torch.inference_mode():
        _, z256_h = dense_world_forward_with_latent(wm, w256_h)

    # Upsample realistic 3D Frostmarsh world to 512x512x4 (1,048,576 voxels!) and run real WorldModel
    w512_h = F.interpolate(w256, size=(512, 512), mode="bilinear", align_corners=False).half().contiguous()
    with torch.inference_mode():
        _, z512_h = dense_world_forward_with_latent(wm, w512_h)

    # Construct 12 realistic bots spanning all 4 layers & acoustic regimes (clear, wall-occluded, waterfall, L0 cavern)
    wf_x, wf_y, _ = lm256["waterfall_positions"][0]
    oil_x, oil_y, _ = lm256["oil_positions"][0]
    fire_x, fire_y, _ = lm256["wildfire_positions"][0]
    ice_x, ice_y, _ = lm256["ice_bridge_pos"]
    nb_x, nb_y, _ = lm256["north_skybridge"]

    demo_bots = [
        # Team 0 (Human's friendly squad) — spanning close open-air, mid-range, waterfall roar, basalt wall, & L0 cavern!
        {"id": 1, "name": "Pyra",    "team": 0, "cls": "Pyromancer",     "x": 0.48, "y": 0.56, "layer": 1, "hp": 95.0},
        {"id": 2, "name": "Glacius", "team": 0, "cls": "Cryo-Scout",     "x": 0.52, "y": 0.44, "layer": 1, "hp": 88.0},
        {"id": 3, "name": "Kael",    "team": 0, "cls": "Vanguard",       "x": nb_x / 256.0, "y": nb_y / 256.0, "layer": 2, "hp": 100.0},
        {"id": 4, "name": "Mire",    "team": 0, "cls": "Shadow-Flanker", "x": wf_x / 256.0, "y": wf_y / 256.0, "layer": 1, "hp": 72.0},
        {"id": 5, "name": "Bram",    "team": 0, "cls": "Vanguard",       "x": 0.16, "y": 0.72, "layer": 0, "hp": 84.0},
        {"id": 6, "name": "Nyx",     "team": 0, "cls": "Shadow-Flanker", "x": oil_x / 256.0, "y": oil_y / 256.0, "layer": 1, "hp": 91.0},
        # Team 1 (Enemy squad)
        {"id": 7, "name": "Vex",     "team": 1, "cls": "Pyromancer",     "x": fire_x / 256.0, "y": fire_y / 256.0, "layer": 1, "hp": 78.0},
        {"id": 8, "name": "Rime",    "team": 1, "cls": "Cryo-Scout",     "x": ice_x / 256.0, "y": ice_y / 256.0, "layer": 1, "hp": 90.0},
        {"id": 9, "name": "Cinder",  "team": 1, "cls": "Pyromancer",     "x": 0.56, "y": 0.53, "layer": 1, "hp": 64.0},
        {"id": 10,"name": "Sable",   "team": 1, "cls": "Shadow-Flanker", "x": 0.80, "y": 0.76, "layer": 1, "hp": 100.0},
        {"id": 11,"name": "Orin",    "team": 1, "cls": "Vanguard",       "x": 0.74, "y": 0.24, "layer": 2, "hp": 96.0},
        {"id": 12,"name": "Dusk",    "team": 1, "cls": "Cryo-Scout",     "x": 0.24, "y": 0.28, "layer": 1, "hp": 82.0},
    ]

    listener_pos = (0.50, 0.50, 1)

    # Benchmark step_live() latency on BOTH 256x256x4 and 512x512x4 (with and without human_chat)
    def _bench_live(w_t: torch.Tensor, z_t: torch.Tensor, chat: dict | None, warmup: int = 15, iters: int = 80) -> dict[str, float]:
        for _ in range(warmup):
            engine.step_live(w_t, z_t, demo_bots, listener_pos, human_chat=chat)
        torch.cuda.synchronize()
        wall_ms = []
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            t_s = time.perf_counter()
            starts[i].record()
            engine.step_live(w_t, z_t, demo_bots, listener_pos, human_chat=chat)
            ends[i].record()
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - t_s) * 1000.0)
        gpu_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return {
            "gpu_p50_ms": round(float(np.median(gpu_ms)), 3),
            "gpu_p95_ms": round(float(np.percentile(gpu_ms, 95)), 3),
            "wall_p50_ms": round(float(np.median(wall_ms)), 3),
            "wall_p95_ms": round(float(np.percentile(wall_ms, 95)), 3),
        }

    sample_chat = {
        "text": "ignite oil at central spire",
        "x": 0.50,
        "y": 0.52,
        "layer": 1,
        "volume": 0.78,
        "team": 0,
    }
    lat_256_speech = _bench_live(w256_h, z256_h, None)
    lat_256_chat = _bench_live(w256_h, z256_h, sample_chat)
    lat_512_speech = _bench_live(w512_h, z512_h, None)
    lat_512_chat = _bench_live(w512_h, z512_h, sample_chat)

    print(
        f"[LiveAcousticEngine] 256x256x4: speech={lat_256_speech['wall_p50_ms']:.3f}ms, with_chat={lat_256_chat['wall_p50_ms']:.3f}ms | "
        f"512x512x4: speech={lat_512_speech['wall_p50_ms']:.3f}ms, with_chat={lat_512_chat['wall_p50_ms']:.3f}ms"
    )

    # Verify the 3 required real human chat commands + diverse SNR degradation regimes!
    command_tests = [
        {"text": "ignite oil at central spire", "x": 0.50, "y": 0.52, "layer": 1, "volume": 0.80, "team": 0},
        {"text": "freeze north bridge",         "x": 0.50, "y": 0.48, "layer": 1, "volume": 0.72, "team": 0},
        {"text": "retreat to sanctuary",        "x": 0.46, "y": 0.54, "layer": 1, "volume": 0.35, "team": 0},
    ]
    human_cmd_results = []
    for cmd in command_tests:
        res = engine.step_live(w256_h, z256_h, demo_bots, listener_pos, human_chat=cmd)
        human_cmd_results.append(
            {
                "command": cmd,
                "decoded_intent": res["human_intent"],
                "overrides_count": len(res["human_overrides"]),
                "human_overrides": res["human_overrides"],
                "human_replies": res["human_replies"],
                "enemy_eavesdropped": res["enemy_eavesdropped"],
            }
        )
        print(f"\n  Human cmd: \"{cmd['text']}\" -> intent={res['human_intent']['intent']} "
              f"(elem={res['human_intent']['elem_name']}, ability={res['human_intent']['primary_ability']})")
        for r in res["human_replies"]:
            print(f"    -> {r['bot_name']:8s} ({r['snr_db']:+6.1f} dB, {r['tier']:9s}): {r['reply_text']}")

    # Verify Bot-to-Bot emergent speech bubbles across 4 SNR listener placements (clear, partial, muffled, inaudible)
    snr_listener_probes = [
        ("near_oil_clear",      (oil_x / 256.0 + 0.02, oil_y / 256.0 + 0.02, 1)),
        ("mid_valley_partial",  (0.50, 0.50, 1)),
        ("behind_waterfall_lo", (wf_x / 256.0, wf_y / 256.0, 1)),
        ("deep_cavern_l0",      (0.86, 0.14, 0)),
    ]
    live_bubble_samples = {}
    for label, lpos in snr_listener_probes:
        out_b = engine.step_live(w256_h, z256_h, demo_bots, lpos, human_chat=None)
        live_bubble_samples[label] = {
            "listener_pos": list(lpos),
            "ambient_stats": out_b["ambient_stats"],
            "speech_bubbles": out_b["speech_bubbles"],
        }
        print(f"\n  Listener [{label} @ {lpos}]:")
        for b in out_b["speech_bubbles"]:
            print(f"    sym={b['symbol_id']:2d} | {b['text']}")

    # Also confirm 512x512x4 live bubble + human chat output parity
    out_512_sample = engine.step_live(w512_h, z512_h, demo_bots, listener_pos, human_chat=command_tests[1])

    report = {
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "artifacts": {
            "sparse_world_runner": str(ROOT / "sparse_world_runner.py"),
            "live_acoustics": str(ROOT / "live_acoustics.py"),
            "checkpoint": str(DEFAULT_CKPT_PATH),
            "checkpoint_bytes": int(DEFAULT_CKPT_PATH.stat().st_size),
        },
        "deliverable1_sparse_world_runner": sparse_verify,
        "deliverable2_live_acoustics": {
            "training_meta": train_meta,
            "latency_benchmarks_ms": {
                "256x256x4_speech_only": lat_256_speech,
                "256x256x4_with_human_chat": lat_256_chat,
                "512x512x4_speech_only": lat_512_speech,
                "512x512x4_with_human_chat": lat_512_chat,
            },
            "under_1_5ms_target": {
                "256x256x4_wall_p50_pass": lat_256_chat["wall_p50_ms"] < 1.5,
                "512x512x4_wall_p50_pass": lat_512_chat["wall_p50_ms"] < 1.5,
            },
            "human_command_verifications": human_cmd_results,
            "bot_speech_snr_samples": live_bubble_samples,
            "grid_512x512x4_live_sample": {
                "grid": out_512_sample["grid"],
                "ambient_stats": out_512_sample["ambient_stats"],
                "speech_bubbles": out_512_sample["speech_bubbles"],
                "human_intent": out_512_sample["human_intent"],
                "human_overrides_count": len(out_512_sample["human_overrides"]),
                "human_replies": out_512_sample["human_replies"],
            },
        },
    }

    out_path = Path(out_json)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved full verification report to {out_path} OK!")
    return report


if __name__ == "__main__":
    run_full_verification(retrain=False)
