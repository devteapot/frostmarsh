"""
Frostmarsh Continuous `Δt`-Conditioned 3D World Model with Structural Stress & Solid
Phase-Transmutation (`WorldModelDT`, ~132K params).

Extends `WorldModel3D` v4 (`[L, 18, H, W]` + `1x1x3` cross-layer mixing + `out_z_flux [L, 5, H, W]`)
with:
  1. Continuous per-voxel time-step conditioning `local_dt(l, y, x) in [0.0, 4.0]` (accepted either
     as the 19th input channel `[..., 19, H, W]` or as an explicit `local_dt` tensor/scalar):
       - Multi-frequency sinusoidal + log-dt basis (`encode_dt_features`, identically 0 at `dt = 1.0`)
       - Per-block spatial FiLM modulation `(1 + scale_i) * h + shift_i` into every `ResConvLayer`
       - Direct `Δt`-conditioned vertical interface & horizontal rate residual modulation
       - Exact face-symmetric multiplicative scaling of all transport/heat fluxes by
         `dt_face = min(dt_here, dt_neighbor)` and cell reactions by `dt_cell`, so:
           * `local_dt == 0.0` yields EXACT zero-change stasis (`0.00e+00` everywhere) by construction
           * Spatially varying `local_dt(l, y, x)` (e.g., 0.15x stasis dome / 3.0x rapid-aging zone)
             preserves exact finite-volume mass & energy conservation across arbitrary time-dilation boundaries!
           * Warm-starting from `world_model_3d_v4.pt` starts with EXACT `Δt = 1.0` parity (`0.00e+00` diff)!
  2. Cross-layer `struct_head` (`out_struct [L, 4, H, W]`):
       - 0: `structural_stress`     in [0, 1]  (thermal & span-load failure risk on elevated bridges/ice)
       - 1: `collapse_z_flux`       >= 0       (downward L2->L1 fractured rubble collapse rate, scaled by `dt_z`)
       - 2: `ice_rampart_integrity` in [0, 1]  (solid walkable/blocking cryo ice rampart barrier strength)
       - 3: `channel_carve_rate`    >= 0       (thermal beam ice-dam carving / channel melt rate, scaled by `dt`)
  3. Exact shrinking-halo active-column tiled forward (`forward_active_columns`) for adaptive
     multi-rate temporal sparsity (active combat zone at `Δt=1.0` every tick + far-field at `Δt=4.0`
     every 4th tick).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RepConv3x3, ResConvLayer
from world_model_3d import (
    Z_FLUX_CH,
    Z_FLUX_NAMES,
    ZF_HEAT,
    ZF_OIL,
    ZF_SMOKE,
    ZF_STEAM,
    ZF_WATER,
)

STRUCT_CH = 4
ST_STRESS = 0
ST_COLLAPSE = 1
ST_ICE_RAMPART = 2
ST_CHANNEL_CARVE = 3

STRUCT_NAMES = (
    "structural_stress",
    "collapse_z_flux",
    "ice_rampart_integrity",
    "channel_carve_rate",
)


def encode_dt_features(local_dt: torch.Tensor) -> torch.Tensor:
    """
    Encode `local_dt` [N, 1, H, W] into 8 smooth sinusoidal + log-dt basis channels.
    Every channel is IDENTICALLY 0.0 when `local_dt == 1.0`, ensuring exact warm-start parity!
    """
    dt = local_dt.clamp(min=0.0, max=4.5)
    u = torch.log(dt.clamp(min=0.05))  # 0.0 at dt=1.0; ~[-3.0, +1.5]
    half_pi_u = (0.5 * math.pi) * u
    return torch.cat(
        [
            0.5 * u,
            (dt - 1.0) / 3.0,
            torch.sin(half_pi_u),
            torch.cos(half_pi_u) - 1.0,
            torch.sin(2.0 * half_pi_u),
            torch.cos(2.0 * half_pi_u) - 1.0,
            torch.sin(4.0 * half_pi_u),
            torch.cos(4.0 * half_pi_u) - 1.0,
        ],
        dim=1,
    )


def build_dt_input_state(state_18: torch.Tensor, local_dt: torch.Tensor | float = 1.0) -> torch.Tensor:
    """Concatenate `local_dt` as the 19th spatial channel onto a 4D `[L,18,H,W]` or 5D `[B,L,18,H,W]` state."""
    if isinstance(local_dt, (int, float)):
        dt_t = torch.full(
            (*state_18.shape[:-3], 1, state_18.shape[-2], state_18.shape[-1]),
            float(local_dt),
            dtype=state_18.dtype,
            device=state_18.device,
        )
    else:
        dt_t = local_dt.to(dtype=state_18.dtype, device=state_18.device)
        if dt_t.ndim == state_18.ndim - 1:
            dt_t = dt_t.unsqueeze(-3)
        dt_t = dt_t.expand(*state_18.shape[:-3], 1, state_18.shape[-2], state_18.shape[-1])
    return torch.cat([state_18, dt_t], dim=-3)


class WorldModelDT(nn.Module):
    """
    Continuous `Δt`-Conditioned True 3D World Model with Structural Stress & Phase Transmutation.
    ~132K parameters (warm-startable with 0.00e+00 initial diff from `world_model_3d_v4.pt`).
    """

    def __init__(
        self,
        in_ch: int = 18,
        out_ch: int = 17,
        width: int = 48,
        num_res: int = 5,
        z_flux_ch: int = Z_FLUX_CH,
        struct_ch: int = STRUCT_CH,
        dt_dim: int = 10,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.width = width
        self.num_res = num_res
        self.z_flux_ch = z_flux_ch
        self.struct_ch = struct_ch
        self.dt_dim = dt_dim

        # 1. Exact WorldModel3D v4 spatial + 3D cross-layer backbone (123,766 params)
        self.stem = RepConv3x3(in_ch, width)
        self.blocks = nn.ModuleList([ResConvLayer(width) for _ in range(num_res)])
        self.head = nn.Conv2d(width, out_ch, kernel_size=1, bias=True)

        scale = torch.ones((1, out_ch, 1, 1), dtype=torch.float32)
        scale[:, 8:10] = 10.0
        self.register_buffer("out_scale", scale)

        with torch.no_grad():
            self.head.weight.mul_(0.2)
            self.head.bias.zero_()
            if out_ch >= 17:
                self.head.bias[10:17].fill_(-2.0)

        self.vert_conv = nn.Conv3d(width, width, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True)
        self.vert_gamma = nn.Parameter(torch.zeros((1, width, 1, 1, 1), dtype=torch.float32))
        with torch.no_grad():
            self.vert_conv.weight.mul_(0.15)
            self.vert_conv.bias.zero_()

        self.vert_lo_in = nn.Conv2d(in_ch, 32, kernel_size=1, bias=True)
        self.vert_hi_in = nn.Conv2d(in_ch, 32, kernel_size=1, bias=False)
        self.vert_lat_in = nn.Conv2d(width, 32, kernel_size=1, bias=False)
        self.vert_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, z_flux_ch, kernel_size=1, bias=True),
        )

        z_scale = torch.ones((1, z_flux_ch, 1, 1), dtype=torch.float32)
        if z_flux_ch >= 5:
            z_scale[:, ZF_HEAT] = 10.0
        self.register_buffer("z_out_scale", z_scale)

        with torch.no_grad():
            self.vert_head[-1].weight.mul_(0.05)
            self.vert_head[-1].bias.zero_()

        # 2. Continuous Δt sinusoidal/log-dt FiLM conditioner (bias=False -> EXACT identity at dt == 1.0!)
        self.dt_mlp = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, dt_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        # Per-block FiLM (scale, shift) for all `num_res` residual blocks
        self.dt_film = nn.Conv2d(dt_dim, num_res * 2 * width, kernel_size=1, bias=False)
        # Direct nonlinear rate correction on horizontal head for multi-step Δt saturation
        self.dt_head_mod = nn.Conv2d(dt_dim, out_ch, kernel_size=1, bias=False)
        # Vertical interface Δt modulation
        self.dt_z_mod = nn.Conv2d(dt_dim, 32, kernel_size=1, bias=False)

        with torch.no_grad():
            self.dt_film.weight.zero_()
            self.dt_head_mod.weight.zero_()
            self.dt_z_mod.weight.zero_()

        # 3. Structural Stress & Solid Phase-Transmutation Head (reads local state + below-layer support + 3D latent + dt)
        self.struct_x_in = nn.Conv2d(in_ch, 24, kernel_size=1, bias=True)
        self.struct_below_in = nn.Conv2d(in_ch, 24, kernel_size=1, bias=False)
        self.struct_lat_in = nn.Conv2d(width, 24, kernel_size=1, bias=False)
        self.struct_dt_in = nn.Conv2d(dt_dim, 24, kernel_size=1, bias=False)
        self.struct_head = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(24, struct_ch, kernel_size=1, bias=True),
        )
        with torch.no_grad():
            self.struct_head[-1].weight.mul_(0.15)
            self.struct_head[-1].bias.zero_()
            self.struct_head[-1].bias[ST_STRESS] = -1.0
            self.struct_head[-1].bias[ST_COLLAPSE] = -2.0
            self.struct_head[-1].bias[ST_ICE_RAMPART] = -1.0
            self.struct_head[-1].bias[ST_CHANNEL_CARVE] = -2.0

    def load_from_v4_state_dict(self, v4_sd: dict[str, torch.Tensor]) -> int:
        """Warm-start all spatial & 3D cross-layer weights from a `WorldModel3D` v4 checkpoint."""
        own = self.state_dict()
        loaded = 0
        for k, v in v4_sd.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                loaded += 1
        assert loaded >= len(v4_sd), f"Expected all {len(v4_sd)} v4 keys to match, matched {loaded}"
        with torch.no_grad():
            self.dt_film.weight.zero_()
            self.dt_head_mod.weight.zero_()
            self.dt_z_mod.weight.zero_()
        return loaded

    def _split_state_and_dt(
        self,
        state: torch.Tensor,
        local_dt: torch.Tensor | float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Normalize input into 5D `x5 [B, L, 18, H, W]` and `dt5 [B, L, 1, H, W]`."""
        is_4d = state.ndim == 4
        if is_4d:
            s5 = state.unsqueeze(0)
        elif state.ndim == 5:
            s5 = state
        else:
            raise ValueError(f"Expected 4D [L,C,H,W] or 5D [B,L,C,H,W] state, got {tuple(state.shape)}")

        B, L, C, H, W = s5.shape
        if C == self.in_ch + 1:
            x5 = s5[:, :, : self.in_ch]
            dt5 = s5[:, :, self.in_ch : self.in_ch + 1]
            if local_dt is not None:
                dt5 = self._broadcast_dt(local_dt, B, L, H, W, s5.dtype, s5.device)
        elif C == self.in_ch:
            x5 = s5
            if local_dt is None:
                dt5 = torch.ones((B, L, 1, H, W), dtype=s5.dtype, device=s5.device)
            else:
                dt5 = self._broadcast_dt(local_dt, B, L, H, W, s5.dtype, s5.device)
        else:
            raise ValueError(f"Expected {self.in_ch} or {self.in_ch + 1} channels, got {C}")

        dt5 = dt5.clamp(min=0.0)
        return x5, dt5, is_4d

    @staticmethod
    def _broadcast_dt(
        local_dt: torch.Tensor | float,
        B: int,
        L: int,
        H: int,
        W: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(local_dt, (int, float)):
            return torch.full((B, L, 1, H, W), float(local_dt), dtype=dtype, device=device)
        t = local_dt.to(dtype=dtype, device=device)
        if t.ndim == 0:
            return t.view(1, 1, 1, 1, 1).expand(B, L, 1, H, W)
        if t.ndim == 2:  # [H, W]
            return t.view(1, 1, 1, H, W).expand(B, L, 1, H, W)
        if t.ndim == 3:  # [L, H, W] or [1, H, W]
            return t.view(1, t.shape[0], 1, H, W).expand(B, L, 1, H, W)
        if t.ndim == 4:  # [L, 1, H, W] or [B, 1, H, W]
            if t.shape[0] == L and B == 1:
                return t.unsqueeze(0).expand(B, L, 1, H, W)
            if t.shape[0] == B:
                return t.unsqueeze(1).expand(B, L, 1, H, W)
            return t.unsqueeze(0).expand(B, L, 1, H, W)
        if t.ndim == 5:
            return t.expand(B, L, 1, H, W)
        raise ValueError(f"Unsupported local_dt shape {tuple(t.shape)}")

    @staticmethod
    def _scale_horiz_by_dt(horiz_rate: torch.Tensor, dt_2d: torch.Tensor) -> torch.Tensor:
        """
        Multiply face fluxes by symmetric face `dt_face = min(dt_here, dt_nb)` and cell reactions by `dt`.
        Guarantees:
          - `dt == 0` -> exact 0.0 stasis across every cell and boundary face!
          - Symmetric face scaling -> exact mass & energy conservation even across sharp time-dilation walls!
        """
        dt_e_nb = F.pad(dt_2d[:, :, :, 1:], (0, 1, 0, 0), mode="replicate")
        dt_s_nb = F.pad(dt_2d[:, :, 1:, :], (0, 0, 0, 1), mode="replicate")
        dt_e = torch.minimum(dt_2d, dt_e_nb)
        dt_s = torch.minimum(dt_2d, dt_s_nb)

        return torch.cat(
            [
                horiz_rate[:, 0:4] * dt_e,
                horiz_rate[:, 4:8] * dt_s,
                horiz_rate[:, 8:9] * dt_e,
                horiz_rate[:, 9:10] * dt_s,
                horiz_rate[:, 10:17] * dt_2d,
            ],
            dim=1,
        )

    def _compute_struct_head(
        self,
        x5: torch.Tensor,
        z_2d: torch.Tensor,
        dt_emb: torch.Tensor,
        dt5: torch.Tensor,
    ) -> torch.Tensor:
        B, L, C_in, H, W = x5.shape
        x_2d = x5.reshape(B * L, C_in, H, W)

        # Build below-layer state `x_below` for each layer z (for z=0, synthetic solid bedrock ground)
        if L > 1:
            ground = torch.zeros((B, 1, C_in, H, W), dtype=x5.dtype, device=x5.device)
            ground[:, :, 4] = 1.0   # rock = 1000 (solid bedrock below L0)
            ground[:, :, 15] = 1.0  # rock_wall = 1.0
            x_below = torch.cat([ground, x5[:, :-1]], dim=1).reshape(B * L, C_in, H, W)
        else:
            x_below = torch.zeros_like(x_2d)
            x_below[:, 4] = 1.0
            x_below[:, 15] = 1.0

        sh = F.gelu(
            self.struct_x_in(x_2d)
            + self.struct_below_in(x_below)
            + self.struct_lat_in(z_2d)
            + self.struct_dt_in(dt_emb)
        )
        s_raw = self.struct_head(sh).view(B, L, self.struct_ch, H, W)

        # Collapse occurs strictly downward from elevated layers z >= 1 into z - 1, scaled by min(dt[z], dt[z-1])
        if L > 1:
            dt_below = torch.zeros_like(dt5)
            dt_below[:, 1:] = torch.minimum(dt5[:, 1:], dt5[:, :-1])
        else:
            dt_below = torch.zeros_like(dt5)

        stress = torch.sigmoid(s_raw[:, :, ST_STRESS : ST_STRESS + 1])
        collapse = F.softplus(s_raw[:, :, ST_COLLAPSE : ST_COLLAPSE + 1]) * dt_below
        ice_rampart = torch.sigmoid(s_raw[:, :, ST_ICE_RAMPART : ST_ICE_RAMPART + 1])
        carve = F.softplus(s_raw[:, :, ST_CHANNEL_CARVE : ST_CHANNEL_CARVE + 1]) * dt5
        return torch.cat([stress, collapse, ice_rampart, carve], dim=2)

    def forward(
        self,
        state: torch.Tensor,
        local_dt: torch.Tensor | float | None = None,
        return_z_flux: bool = True,
        return_struct: bool = True,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Forward pass supporting 4D `[L, 18|19, H, W]` or 5D `[B, L, 18|19, H, W]`.

        Returns (depending on flags):
          - `horiz`:      `[..., 17, H, W]` (flux_e, flux_s, heat_e, heat_s, rxn scaled by local_dt)
          - `out_z_flux`: `[..., 5, H, W]`  (vertical gravity/buoyancy/heat fluxes scaled by dt_z)
          - `out_struct`: `[..., 4, H, W]`  (structural_stress, collapse_z_flux, ice_rampart_integrity, channel_carve_rate)
          - `z_world`:    `[..., 48, H, W]` (3D-mixed latent representation)
        """
        x5, dt5, is_4d = self._split_state_and_dt(state, local_dt)
        B, L, C_in, H, W = x5.shape
        x_2d = x5.reshape(B * L, C_in, H, W)
        dt_2d = dt5.reshape(B * L, 1, H, W)

        # Continuous Δt sinusoidal/log-dt FiLM embeddings
        dt_feat = encode_dt_features(dt_2d)
        dt_emb = self.dt_mlp(dt_feat)  # [B*L, dt_dim, H, W]
        film_all = self.dt_film(dt_emb).view(B * L, self.num_res, 2, self.width, H, W)

        # FiLM-modulated 2D spatial residual trunk
        h = self.stem(x_2d)
        for idx, blk in enumerate(self.blocks):
            gamma_f = film_all[:, idx, 0]
            beta_f = film_all[:, idx, 1]
            # Exact identity when gamma_f == 0 and beta_f == 0 (i.e., dt == 1.0)!
            h_mod = h * (1.0 + 0.5 * torch.tanh(gamma_f)) + beta_f
            h = h + blk.gamma * blk.conv(F.gelu(h_mod))

        # 1x1x3 vertical cross-layer mixing layer
        h_3d = h.view(B, L, self.width, H, W).transpose(1, 2)  # [B, width, L, H, W]
        v_mix = self.vert_conv(F.gelu(h_3d))
        h_3d_mixed = h_3d + self.vert_gamma * v_mix
        z_5d = F.gelu(h_3d_mixed).transpose(1, 2)  # [B, L, width, H, W]
        z_2d = z_5d.reshape(B * L, self.width, H, W)

        # Horizontal 17-channel rate prediction + nonlinear Δt correction + exact face-dt scaling
        raw_h = self.head(z_2d) * self.out_scale
        head_mod = self.dt_head_mod(dt_emb)  # 0.0 when dt == 1.0
        # Multiplicative rate modulation in (0.15, 1.85) for sub-linear saturation over large Δt
        rate_mult = 1.0 + 0.85 * torch.tanh(head_mod)
        lin_rate = raw_h[:, :10] * rate_mult[:, :10]
        rxn_rate = F.softplus(raw_h[:, 10:]) * rate_mult[:, 10:]
        horiz_rate = torch.cat([lin_rate, rxn_rate], dim=1)
        horiz_2d = self._scale_horiz_by_dt(horiz_rate, dt_2d)
        horiz = horiz_2d.view(B, L, self.out_ch, H, W)

        out_z_flux: torch.Tensor | None = None
        if return_z_flux:
            if L > 1:
                x_lo = x5[:, :-1].reshape(B * (L - 1), C_in, H, W)
                x_hi = x5[:, 1:].reshape(B * (L - 1), C_in, H, W)
                v_lo = v_mix[:, :, :-1].transpose(1, 2).reshape(B * (L - 1), self.width, H, W)
                dt_z = torch.minimum(dt5[:, :-1], dt5[:, 1:])  # [B, L-1, 1, H, W]
                dt_z_2d = dt_z.reshape(B * (L - 1), 1, H, W)
                dt_z_emb = self.dt_mlp(encode_dt_features(dt_z_2d))
                vh = F.gelu(
                    self.vert_lo_in(x_lo)
                    + self.vert_hi_in(x_hi)
                    + self.vert_lat_in(v_lo)
                    + self.dt_z_mod(dt_z_emb)
                )
                z_if_rate = (self.vert_head(vh) * self.z_out_scale).view(B, L - 1, self.z_flux_ch, H, W)
                z_if = z_if_rate * dt_z
                z_top = torch.zeros((B, 1, self.z_flux_ch, H, W), dtype=z_if.dtype, device=z_if.device)
                out_z_flux = torch.cat([z_if, z_top], dim=1)
            else:
                out_z_flux = torch.zeros((B, L, self.z_flux_ch, H, W), dtype=horiz.dtype, device=horiz.device)

        out_struct: torch.Tensor | None = None
        if return_struct:
            out_struct = self._compute_struct_head(x5, z_2d, dt_emb, dt5)

        z_world = z_5d.contiguous() if return_latent else None

        if is_4d:
            horiz = horiz.squeeze(0)
            if out_z_flux is not None:
                out_z_flux = out_z_flux.squeeze(0)
            if out_struct is not None:
                out_struct = out_struct.squeeze(0)
            if z_world is not None:
                z_world = z_world.squeeze(0)

        ret: list[torch.Tensor] = [horiz]
        if return_z_flux:
            assert out_z_flux is not None
            ret.append(out_z_flux)
        if return_struct:
            assert out_struct is not None
            ret.append(out_struct)
        if return_latent:
            assert z_world is not None
            ret.append(z_world)
        return ret[0] if len(ret) == 1 else tuple(ret)

    def forward_active_columns(
        self,
        patches_x: torch.Tensor,
        patches_dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Exact shrinking-halo valid-conv 3D forward on active tile columns!
        Inputs:
          - `patches_x`:  `[K, L, 18, T + 2*halo, T + 2*halo]` (halo = 1 + num_res = 6)
          - `patches_dt`: `[K, L, 1,  T + 2*halo, T + 2*halo]`
        Returns:
          - `horiz`:      `[K, L, 17, T, T]`
          - `out_z_flux`: `[K, L, 5,  T, T]`
          - `out_struct`: `[K, L, 4,  T, T]`
        Mathematically equivalent to dense `forward()` on every interior cell while skipping
        inactive far-field tiles on ticks 0, 1, 2 of a 4-tick adaptive multi-rate cycle!
        """
        K, L, C_in, Hp, Wp = patches_x.shape
        halo = 1 + self.num_res
        T_h = Hp - 2 * halo
        T_w = Wp - 2 * halo
        assert T_h > 0 and T_w > 0, f"Patch size {(Hp, Wp)} too small for halo={halo}"

        x_2d = patches_x.reshape(K * L, C_in, Hp, Wp)
        dt_2d_full = patches_dt.clamp(min=0.0).reshape(K * L, 1, Hp, Wp)

        # Stem valid conv shrinks spatial dims by 2: Hp -> Hp - 2
        h = F.conv2d(x_2d, self.stem.conv.weight, self.stem.conv.bias, padding=0)
        # Compute dt_emb once on the post-stem [Hp-2, Wp-2] crop and slice inward per residual block
        dt_cur = dt_2d_full[:, :, 1:-1, 1:-1]
        dt_emb_cur = self.dt_mlp(encode_dt_features(dt_cur))
        film_cur = self.dt_film(dt_emb_cur).view(K * L, self.num_res, 2, self.width, Hp - 2, Wp - 2)

        for idx, blk in enumerate(self.blocks):
            gamma_f = film_cur[:, idx, 0]
            beta_f = film_cur[:, idx, 1]
            h_mod = h * (1.0 + 0.5 * torch.tanh(gamma_f)) + beta_f
            conv_out = F.conv2d(F.gelu(h_mod), blk.conv.conv.weight, blk.conv.conv.bias, padding=0)
            h = h[:, :, 1:-1, 1:-1] + blk.gamma * conv_out
            if idx + 1 < self.num_res:
                film_cur = film_cur[:, :, :, :, 1:-1, 1:-1]

        # Crop center [T_h, T_w] of x, dt, and dt_emb (plus 1-pixel east/south margin for dt_face)
        x_int = patches_x[:, :, :, halo:-halo, halo:-halo]  # [K, L, 18, T_h, T_w]
        dt_int5 = patches_dt[:, :, :, halo:-halo, halo:-halo].clamp(min=0.0)
        dt_int2 = dt_int5.reshape(K * L, 1, T_h, T_w)
        # Use true 1-pixel east/south neighbors from the patch halo for exact face dt!
        dt_e_nb = patches_dt[:, :, :, halo:-halo, halo + 1 : Wp - halo + 1].clamp(min=0.0).reshape(K * L, 1, T_h, T_w)
        dt_s_nb = patches_dt[:, :, :, halo + 1 : Hp - halo + 1, halo:-halo].clamp(min=0.0).reshape(K * L, 1, T_h, T_w)
        dt_e = torch.minimum(dt_int2, dt_e_nb)
        dt_s = torch.minimum(dt_int2, dt_s_nb)

        dt_emb_int = dt_emb_cur[:, :, self.num_res : -self.num_res, self.num_res : -self.num_res]

        # 1x1x3 vertical cross-layer mixing on the K active tile columns
        h_3d = h.view(K, L, self.width, T_h, T_w).transpose(1, 2)
        v_mix = self.vert_conv(F.gelu(h_3d))
        h_3d_mixed = h_3d + self.vert_gamma * v_mix
        z_5d = F.gelu(h_3d_mixed).transpose(1, 2)
        z_2d = z_5d.reshape(K * L, self.width, T_h, T_w)

        raw_h = self.head(z_2d) * self.out_scale
        rate_mult = 1.0 + 0.85 * torch.tanh(self.dt_head_mod(dt_emb_int))
        horiz_rate = torch.cat([raw_h[:, :10] * rate_mult[:, :10], F.softplus(raw_h[:, 10:]) * rate_mult[:, 10:]], dim=1)
        horiz_2d = torch.cat(
            [
                horiz_rate[:, 0:4] * dt_e,
                horiz_rate[:, 4:8] * dt_s,
                horiz_rate[:, 8:9] * dt_e,
                horiz_rate[:, 9:10] * dt_s,
                horiz_rate[:, 10:17] * dt_int2,
            ],
            dim=1,
        )
        horiz = horiz_2d.view(K, L, self.out_ch, T_h, T_w)

        if L > 1:
            x_lo = x_int[:, :-1].reshape(K * (L - 1), C_in, T_h, T_w)
            x_hi = x_int[:, 1:].reshape(K * (L - 1), C_in, T_h, T_w)
            v_lo = v_mix[:, :, :-1].transpose(1, 2).reshape(K * (L - 1), self.width, T_h, T_w)
            dt_z = torch.minimum(dt_int5[:, :-1], dt_int5[:, 1:])
            dt_z_emb = self.dt_mlp(encode_dt_features(dt_z.reshape(K * (L - 1), 1, T_h, T_w)))
            vh = F.gelu(
                self.vert_lo_in(x_lo)
                + self.vert_hi_in(x_hi)
                + self.vert_lat_in(v_lo)
                + self.dt_z_mod(dt_z_emb)
            )
            z_if = (self.vert_head(vh) * self.z_out_scale).view(K, L - 1, self.z_flux_ch, T_h, T_w) * dt_z
            z_top = torch.zeros((K, 1, self.z_flux_ch, T_h, T_w), dtype=z_if.dtype, device=z_if.device)
            out_z_flux = torch.cat([z_if, z_top], dim=1)
        else:
            out_z_flux = torch.zeros((K, L, self.z_flux_ch, T_h, T_w), dtype=horiz.dtype, device=horiz.device)

        out_struct = self._compute_struct_head(x_int, z_2d, dt_emb_int, dt_int5)
        return horiz, out_z_flux, out_struct


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
