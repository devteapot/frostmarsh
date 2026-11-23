"""
Frostmarsh Neural World Model & Phase-3 Ecosystem Neural Field Architectures.

WorldModel:
- Pure Conv2d bounded-RF 6x(3x3) Residual CNN (width=48, GELU, replicate pad -> RF=13).
- ~113k parameters, accepts arbitrary dynamic [B, 18, H, W] -> [B, 17, H, W].
- Softplus applied to rxn_extent channels 10..17 inside forward().

EcosystemModel (Phase 3 / G5-B):
- Pure Conv2d bounded-RF 4x(3x3) Residual CNN (width=32, GELU, replicate pad -> RF=9).
- Accepts [B, 8, H, W] -> [B, 4, H, W] predicting critter density, fear, forage depletion, herd momentum.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepConv3x3(nn.Module):
    """3x3 Conv2d with explicit replicate padding (exports cleanly to ONNX Pad mode='edge')."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return self.conv(x)


class ResConvLayer(nn.Module):
    """Single 3x3 residual layer with learnable residual scale for stable deep training."""

    def __init__(self, width: int):
        super().__init__()
        self.conv = RepConv3x3(width, width)
        self.gamma = nn.Parameter(torch.full((1, width, 1, 1), 0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.conv(F.gelu(x))


class WorldModel(nn.Module):
    """
    Bounded-RF 6x(3x3) Residual CNN (RF = 1 + 6*2 = 13).
    Input:  state  [B, 18, H, W] float32
    Output: deltas [B, 17, H, W] float32
      0..4:   flux_e[water, steam, smoke, oil] in mass / 1000.0
      4..8:   flux_s[water, steam, smoke, oil] in mass / 1000.0
      8..10:  heat_e, heat_s in 1e6 energy units
      10..17: rxn_extent[0..7] (softplus, in 100.0 units)
    """

    def __init__(self, in_ch: int = 18, out_ch: int = 17, width: int = 48, num_res: int = 5):
        super().__init__()
        self.stem = RepConv3x3(in_ch, width)
        self.blocks = nn.ModuleList([ResConvLayer(width) for _ in range(num_res)])
        self.head = nn.Conv2d(width, out_ch, kernel_size=1, bias=True)

        # Fixed per-channel output scale so internal activations stay O(1)
        # flux_e/s: 1.0, heat_e/s: 10.0, rxn: 1.0 (before softplus)
        scale = torch.ones((1, out_ch, 1, 1), dtype=torch.float32)
        scale[:, 8:10] = 10.0
        self.register_buffer("out_scale", scale)

        # Initialize reaction head bias moderately negative (-2.0) so softplus starts small without vanishing gradients
        with torch.no_grad():
            self.head.weight.mul_(0.2)
            self.head.bias.zero_()
            self.head.bias[10:17].fill_(-2.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.stem(state)
        for blk in self.blocks:
            h = blk(h)
        raw = self.head(F.gelu(h)) * self.out_scale
        # Split linear transport/heat channels (0..10) and non-negative reaction channels (10..17)
        lin = raw[:, :10]
        rxn = F.softplus(raw[:, 10:])
        return torch.cat([lin, rxn], dim=1)


class EcosystemModel(nn.Module):
    """
    Phase 3 Living Ecosystem 10 Hz Conservative Neural Field (RF=13, 6x 3x3 Residual CNN).
    Input:  eco_state [B, 8, H, W] float32
      0: critter_density
      1: fear (0..1)
      2: prev_forage_depletion (in 100.0 plant mass units)
      3: herd_momentum (density * speed)
      4: plant_mass / 1000.0
      5: deep_water_barrier (1 if water>=350 and ice<700 else 0)
      6: hazard_field (fire / smoke / high T)
      7: rock_wall (1 if rock>=500 else 0)
    Output: eco_next [B, 4, H, W] float32
      0: next_critter_density (conservatively advected, >=0, zero wall/water penetration)
      1: next_fear (sigmoid in [0, 1])
      2: forage_depletion (in 100.0 plant mass units, bounded by density * plant)
      3: next_herd_momentum (density * predicted speed, strictly bounded)
    """

    def __init__(self, in_ch: int = 8, out_ch: int = 4, width: int = 32, num_res: int = 5):
        super().__init__()
        self.stem = RepConv3x3(in_ch, width)
        self.blocks = nn.ModuleList([ResConvLayer(width) for _ in range(num_res)])
        # Internal head predicts 5 fields: (flux_e_rate, flux_s_rate, fear_logit, graze_logit, speed_logit)
        self.head = nn.Conv2d(width, 5, kernel_size=1, bias=True)
        with torch.no_grad():
            self.head.weight.mul_(0.2)
            self.head.bias.zero_()
            self.head.bias[2].fill_(-2.0)  # fear starts low
            self.head.bias[3].fill_(-1.5)  # grazing starts modest
            self.head.bias[4].fill_(-1.0)  # speed starts modest

    def forward(self, eco_state: torch.Tensor) -> torch.Tensor:
        rho = torch.clamp(eco_state[:, 0:1], min=0.0)
        plant = torch.clamp(eco_state[:, 4:5], min=0.0)
        water_bar = eco_state[:, 5:6]
        rock_wall = eco_state[:, 7:8]
        open_cell = torch.clamp(1.0 - water_bar - rock_wall, 0.0, 1.0)

        h = self.stem(eco_state)
        for blk in self.blocks:
            h = blk(h)
        raw = self.head(F.gelu(h))

        # Conservative edge fluxes for critter density (zero into walls / unfrozen deep water)
        open_e = open_cell * F.pad(open_cell[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
        open_s = open_cell * F.pad(open_cell[:, :, 1:, :], (0, 0, 0, 1), value=0.0)

        rho_e_nb = F.pad(rho[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
        rho_s_nb = F.pad(rho[:, :, 1:, :], (0, 0, 0, 1), value=0.0)

        rate_e = torch.tanh(raw[:, 0:1]) * 0.65
        rate_s = torch.tanh(raw[:, 1:2]) * 0.65
        fe = torch.where(rate_e >= 0.0, rate_e * rho, rate_e * rho_e_nb) * open_e
        fs = torch.where(rate_s >= 0.0, rate_s * rho, rate_s * rho_s_nb) * open_s

        # Donor-budget positivity clamp -> exact critter conservation + non-negativity
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

        return torch.cat([next_rho, next_fear, forage, next_momentum], dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
