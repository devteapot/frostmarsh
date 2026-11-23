#!/usr/bin/env python3
"""
Production Shrinking-Halo Sparse Active-Column 3D WorldModel Runner (`SparseWorld3DRunner`).

Drop-in wrapper around Frostmarsh's true 3D cross-layer `WorldModel3D` v4 (123.8K params:
6x(3x3) 2D spatial residual trunk with RF=13 / exact halo=6, 1x1x3 vertical cross-layer
mixing conv `vert_conv`, horizontal 17-ch head, and 5-ch staggered vertical interface head
`out_z_flux` on L-1 active interfaces):

  - Maintains persistent caches on device:
      self._last_horiz  : [L,   17, H, W]  (horizontal physics flux/heat/reaction deltas)
      self._last_z_flux : [L-1,  5, H, W]  (staggered vertical interface fluxes: water, oil, steam, smoke, heat_z)
      self._last_z      : [L,   48, H, W]  (shared 3D-mixed WorldModel latent trunk Z_world)
      self._last_x      : [L,   18, H, W]  (previous world state for automatic dirty-column detection)
  - On first call, shape change, non-tile-aligned dims, or active column fraction >= sparse_threshold (42%):
      runs dense `WorldModel3D` and refreshes all caches, returning `(out_horiz, out_z_flux, z_world, 1.0)`.
  - Otherwise:
      1. Computes per-column activity mask `[th, tw]` across ALL L layers (since `vert_conv` couples
         layers vertically along each `(ty, tx)` column).
      2. Zero-copy strided gather of `[K, L, 18, 28, 28]` active column patches (`tile_size + 2*halo`).
      3. 128-column bucketing so cuDNN never re-autotunes on `K +- 1` jitter.
      4. Exact mathematically identical shrinking-halo valid-conv pyramid (`28 -> 26 -> 24 -> 22 -> 20 -> 18 -> 16`)
         followed by exact `1x1x3` vertical mixing + horizontal head + `L-1` vertical interface head
         (CPU FP32 `max_abs_diff < 1e-5` on interior active columns!).
      5. Direct in-place strided scatter into persistent caches with zero full-grid copies.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from world_model_3d import WorldModel3D  # noqa: E402


def dense_world_3d_forward_with_latent(
    wm: WorldModel3D, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact dense `WorldModel3D` forward on `[L, 18, H, W]` returning:
      - `out_horiz` : `[L, 17, H, W]`
      - `out_z_flux`: `[L - 1, 5, H, W]` (active vertical interfaces z <-> z+1)
      - `z_world`   : `[L, 48, H, W]`
    """
    horiz, z_flux_full, z_world = wm(state, return_z_flux=True, return_latent=True)
    L = state.shape[0]
    if L > 1:
        z_flux = z_flux_full[:-1].contiguous()
    else:
        z_flux = z_flux_full[:0].contiguous()
    return horiz, z_flux, z_world


def shrinking_halo_3d_forward_with_latent(
    wm: WorldModel3D,
    patches: torch.Tensor,
    halo: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact mathematically equivalent shrinking-halo `WorldModel3D` pass on active 3D column patches:
      Input:  `patches` `[K, L, 18, T + 2*halo, T + 2*halo]` (typically `[K, 4, 18, 28, 28]`)
      Output:
        - `horiz` : `[K, L, 17, T, T]`
        - `z_flux`: `[K, L - 1, 5, T, T]`
        - `z_5d`  : `[K, L, 48, T, T]`
    """
    K, L, C_in, P, _ = patches.shape
    T = P - 2 * halo
    x_center = patches[:, :, :, halo:-halo, halo:-halo]  # [K, L, C_in, T, T]

    # 1. Exact shrinking-halo 2D spatial residual trunk (28 -> 26 -> 24 -> 22 -> 20 -> 18 -> 16)
    x_2d = patches.reshape(K * L, C_in, P, P)
    h = F.conv2d(x_2d, wm.stem.conv.weight, wm.stem.conv.bias, padding=0)
    for blk in wm.blocks:
        conv_out = F.conv2d(F.gelu(h), blk.conv.conv.weight, blk.conv.conv.bias, padding=0)
        h = h[:, :, 1:-1, 1:-1] + blk.gamma * conv_out

    # 2. Exact 1x1x3 vertical cross-layer mixing layer (purely vertical, zero extra spatial halo!)
    h_3d = h.view(K, L, wm.width, T, T).transpose(1, 2)  # [K, width, L, T, T]
    v_mix = wm.vert_conv(F.gelu(h_3d))
    h_3d_mixed = h_3d + wm.vert_gamma * v_mix
    z_5d = F.gelu(h_3d_mixed).transpose(1, 2)  # [K, L, width, T, T]
    z_2d = z_5d.reshape(K * L, wm.width, T, T)

    # 3. Exact horizontal 17-channel head
    raw_h = wm.head(z_2d) * wm.out_scale
    horiz_2d = torch.cat([raw_h[:, :10], F.softplus(raw_h[:, 10:])], dim=1)
    horiz = horiz_2d.view(K, L, wm.out_ch, T, T)

    # 4. Exact staggered vertical interface head on L-1 active interfaces (purely 1x1 spatially!)
    if L > 1:
        x_lo = x_center[:, :-1].reshape(K * (L - 1), C_in, T, T)
        x_hi = x_center[:, 1:].reshape(K * (L - 1), C_in, T, T)
        v_lo = v_mix[:, :, :-1].transpose(1, 2).reshape(K * (L - 1), wm.width, T, T)
        vh = F.gelu(wm.vert_lo_in(x_lo) + wm.vert_hi_in(x_hi) + wm.vert_lat_in(v_lo))
        z_if = (wm.vert_head(vh) * wm.z_out_scale).view(K, L - 1, wm.z_flux_ch, T, T)
    else:
        z_if = horiz.new_zeros((K, 0, wm.z_flux_ch, T, T))

    return horiz, z_if, z_5d


class SparseWorld3DRunner:
    """
    Production sparse active-column 3D WorldModel v4 runner with persistent device caches
    for horizontal physics deltas (`_last_horiz`), staggered vertical interface fluxes
    (`_last_z_flux`), and shared 3D-mixed latent trunk (`_last_z`).
    """

    def __init__(
        self,
        world_model_3d: WorldModel3D,
        device: torch.device | str,
        use_fp16: bool = True,
        tile_size: int = 16,
        halo: int = 6,
        sparse_threshold: float = 0.42,
        diff_eps: float = 1e-4,
    ) -> None:
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16) and (self.device.type == "cuda")
        self.compute_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.tile_size = int(tile_size)
        self.halo = int(halo)
        self.patch_size = self.tile_size + 2 * self.halo
        self.sparse_threshold = float(sparse_threshold)
        self.diff_eps = float(diff_eps)

        self._required_halo = 1 + len(world_model_3d.blocks)
        if self.halo < self._required_halo:
            raise ValueError(
                f"halo={self.halo} is smaller than WorldModel3D receptive-field radius {self._required_halo}"
            )
        self._extra_halo = self.halo - self._required_halo

        # Keep an isolated eval copy in compute_dtype so caller's model is never mutated
        self.world_model_3d = world_model_3d
        self._wm = copy.deepcopy(world_model_3d).to(device=self.device, dtype=self.compute_dtype).eval()
        for p in self._wm.parameters():
            p.requires_grad_(False)

        self._last_horiz: torch.Tensor | None = None
        self._last_z_flux: torch.Tensor | None = None
        self._last_z: torch.Tensor | None = None
        self._last_x: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear persistent caches (forces dense initialization on next step)."""
        self._last_horiz = None
        self._last_z_flux = None
        self._last_z = None
        self._last_x = None

    def _compute_column_mask(
        self,
        x_cur: torch.Tensor,
        prev_x: torch.Tensor | None,
        dirty_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return boolean active vertical-column mask of shape `[th, tw]` on `self.device`."""
        L, _, H, W = x_cur.shape
        T = self.tile_size
        th, tw = H // T, W // T

        if dirty_mask is not None:
            dm = dirty_mask.to(device=self.device)
            if dm.shape == (th, tw):
                return dm.bool()
            if dm.shape == (L, th, tw):
                return dm.bool().any(dim=0)
            if dm.shape == (H, W):
                pooled = F.max_pool2d(
                    dm.to(dtype=self.compute_dtype).unsqueeze(0).unsqueeze(0),
                    kernel_size=T,
                    stride=T,
                )
                return pooled.squeeze(0).squeeze(0) > 0.5
            if dm.shape == (L, H, W):
                col_px = dm.bool().any(dim=0).to(dtype=self.compute_dtype)
                pooled = F.max_pool2d(
                    col_px.unsqueeze(0).unsqueeze(0),
                    kernel_size=T,
                    stride=T,
                )
                return pooled.squeeze(0).squeeze(0) > 0.5
            raise ValueError(
                f"dirty_mask shape {tuple(dirty_mask.shape)} does not match column grid "
                f"{(th, tw)}, tile grid {(L, th, tw)}, or pixel grid {(H, W)} / {(L, H, W)}"
            )

        ref = prev_x.to(device=self.device, dtype=self.compute_dtype) if prev_x is not None else self._last_x
        assert ref is not None
        d_sp = (x_cur[:, :10] - ref[:, :10]).abs().amax(dim=1) > max(self.diff_eps, 0.015)
        d_tmp = (x_cur[:, 10] - ref[:, 10]).abs() > max(self.diff_eps * 4.0, 0.04)
        has_force = (x_cur[:, 16].abs() > 0.01) | (x_cur[:, 17].abs() > 0.01)
        # Union activity across all L layers so every active column updates all L strata jointly
        diff_col_px = (d_sp | d_tmp | has_force).any(dim=0).to(dtype=self.compute_dtype)  # [H, W]
        pooled = F.max_pool2d(diff_col_px.unsqueeze(0).unsqueeze(0), kernel_size=T, stride=T)  # [1, 1, th, tw]
        return pooled.squeeze(0).squeeze(0) > 0.5

    @torch.inference_mode()
    def step(
        self,
        x: torch.Tensor,
        prev_x: torch.Tensor | None = None,
        dirty_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Advance `WorldModel3D` on `x` (`[L, 18, H, W]`) and return:
            out_horiz   : `[L, 17, H, W]` horizontal physics deltas
            out_z_flux  : `[L - 1, 5, H, W]` staggered vertical interface fluxes
            z_world     : `[L, 48, H, W]` shared 3D-mixed WorldModel latent trunk
            active_frac : fraction of 16x16 vertical columns evaluated (1.0 when dense pass ran)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [L, 18, H, W], got shape {tuple(x.shape)}")

        x_cur = x.to(device=self.device, dtype=self.compute_dtype, non_blocking=(self.device.type == "cuda"))
        L, C, H, W = x_cur.shape
        T = self.tile_size

        need_dense = (
            self._last_horiz is None
            or self._last_z_flux is None
            or self._last_z is None
            or self._last_x is None
            or self._last_x.shape != x_cur.shape
            or (H % T != 0)
            or (W % T != 0)
        )

        if not need_dense:
            assert self._last_horiz is not None
            assert self._last_z_flux is not None
            assert self._last_z is not None
            assert self._last_x is not None

            th, tw = H // T, W // T
            total_cols = th * tw
            col_active = self._compute_column_mask(x_cur, prev_x, dirty_mask)  # [th, tw]
            coords = torch.nonzero(col_active, as_tuple=False)  # [K, 2] -> (ty, tx)
            k = int(coords.shape[0])
            active_frac = float(k) / float(total_cols)

            if k == 0:
                self._last_x.copy_(x_cur)
                return self._last_horiz, self._last_z_flux, self._last_z, 0.0

            if active_frac < self.sparse_threshold:
                ty, tx = coords[:, 0], coords[:, 1]

                # Zero-copy strided unfold view -> gather ONLY the K active [K, L, 18, 28, 28] columns
                padded = F.pad(x_cur, (self.halo, self.halo, self.halo, self.halo), mode="replicate")
                unfolded = padded.unfold(2, self.patch_size, T).unfold(3, self.patch_size, T)
                # unfolded: [L, C, th, tw, P, P] -> permute to [th, tw, L, C, P, P]
                patches = unfolded.permute(2, 3, 0, 1, 4, 5)[ty, tx].contiguous()

                if self._extra_halo > 0:
                    e = self._extra_halo
                    patches = patches[:, :, :, e:-e, e:-e].contiguous()

                # 128-column bucketing so cuDNN never re-autotunes on K +- 1 jitter!
                k_pad = ((k + 127) // 128) * 128
                if k_pad > k:
                    pad_extra = patches.new_zeros(
                        (k_pad - k, L, C, patches.shape[3], patches.shape[4])
                    )
                    patches_run = torch.cat([patches, pad_extra], dim=0)
                else:
                    patches_run = patches

                y_horiz, y_z_flux, y_z = shrinking_halo_3d_forward_with_latent(
                    self._wm, patches_run, halo=self._required_halo
                )
                if k_pad > k:
                    y_horiz = y_horiz[:k]
                    y_z_flux = y_z_flux[:k]
                    y_z = y_z[:k]

                # Direct in-place strided scatter into persistent caches (zero full-grid copy!)
                out_ch = self._last_horiz.shape[1]
                z_ch = self._last_z.shape[1]
                self._last_horiz.view(L, out_ch, th, T, tw, T).permute(2, 4, 0, 1, 3, 5)[ty, tx] = y_horiz
                if L > 1:
                    zf_ch = self._last_z_flux.shape[1]
                    self._last_z_flux.view(L - 1, zf_ch, th, T, tw, T).permute(2, 4, 0, 1, 3, 5)[
                        ty, tx
                    ] = y_z_flux
                self._last_z.view(L, z_ch, th, T, tw, T).permute(2, 4, 0, 1, 3, 5)[ty, tx] = y_z
                self._last_x.copy_(x_cur)
                return self._last_horiz, self._last_z_flux, self._last_z, active_frac

        # Dense fallback / initial cache population
        horiz, z_flux, z_world = dense_world_3d_forward_with_latent(self._wm, x_cur)
        self._last_horiz = horiz.contiguous()
        self._last_z_flux = z_flux.contiguous()
        self._last_z = z_world.contiguous()
        self._last_x = x_cur.clone()
        return self._last_horiz, self._last_z_flux, self._last_z, 1.0


def verify_sparse_world_3d_parity(h: int = 128, w: int = 128, layers: int = 4) -> dict[str, float]:
    """
    Verify on CPU FP32 that sparse interior active columns match dense `WorldModel3D`
    (`max_abs_diff < 1e-5`) across `out_horiz`, `out_z_flux`, and `z_world`!
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    wm = WorldModel3D(in_ch=18, out_ch=17, width=48, num_res=5, z_flux_ch=5).to(device).eval()
    # Give vert_gamma a non-zero value so 3D vertical mixing is actively exercised!
    with torch.no_grad():
        wm.vert_gamma.fill_(0.35)

    prev_x = torch.rand((layers, 18, h, w), dtype=torch.float32, device=device) * 0.5
    prev_x[:, 16:18] = 0.0  # clear live forcing channels in baseline
    next_x = prev_x.clone()

    # Perturb localized interior columns (approx ~20% active columns, well below 42% threshold)
    T = 16
    th, tw = h // T, w // T
    col_mask = torch.zeros((th, tw), dtype=torch.bool, device=device)
    for ty in range(2, min(th - 2, 5)):
        for tx in range(2, min(tw - 2, 6)):
            col_mask[ty, tx] = True
            next_x[:, 0, ty * T : (ty + 1) * T, tx * T : (tx + 1) * T] += 0.25
            next_x[1, 10, ty * T : (ty + 1) * T, tx * T : (tx + 1) * T] += 0.40

    runner = SparseWorld3DRunner(
        wm, device=device, use_fp16=False, tile_size=T, halo=6, sparse_threshold=0.42
    )
    with torch.inference_mode():
        _, _, _, frac0 = runner.step(prev_x)
        sp_h, sp_zf, sp_z, frac1 = runner.step(next_x)
        dn_h, dn_zf, dn_z = dense_world_3d_forward_with_latent(wm, next_x)

    assert frac0 == 1.0, f"Expected initial dense pass (1.0), got {frac0}"
    assert 0.0 < frac1 < 0.42, f"Expected sparse pass (< 0.42), got {frac1}"
    assert tuple(sp_h.shape) == (layers, 17, h, w), f"Bad horiz shape {tuple(sp_h.shape)}"
    assert tuple(sp_zf.shape) == (layers - 1, 5, h, w), f"Bad z_flux shape {tuple(sp_zf.shape)}"
    assert tuple(sp_z.shape) == (layers, 48, h, w), f"Bad z_world shape {tuple(sp_z.shape)}"

    active_px = col_mask.repeat_interleave(T, dim=0).repeat_interleave(T, dim=1)  # [H, W]
    interior = torch.zeros_like(active_px)
    interior[6 : h - 6, 6 : w - 6] = True
    eval_px = active_px & interior

    diff_h = float((sp_h - dn_h).abs()[:, :, eval_px].max().item())
    diff_zf = float((sp_zf - dn_zf).abs()[:, :, eval_px].max().item())
    diff_z = float((sp_z - dn_z).abs()[:, :, eval_px].max().item())

    print(
        f"[SparseWorld3DRunner CPU FP32 Parity] grid={h}x{w}x{layers} active_frac={frac1:.3f} | "
        f"max_abs_diff(horiz={diff_h:.2e}, z_flux={diff_zf:.2e}, z_world={diff_z:.2e})"
    )
    assert diff_h < 1e-5, f"horiz parity failed: {diff_h}"
    assert diff_zf < 1e-5, f"z_flux parity failed: {diff_zf}"
    assert diff_z < 1e-5, f"z_world parity failed: {diff_z}"

    return {
        "active_frac": frac1,
        "max_abs_diff_horiz": diff_h,
        "max_abs_diff_z_flux": diff_zf,
        "max_abs_diff_z_world": diff_z,
    }


if __name__ == "__main__":
    verify_sparse_world_3d_parity()
