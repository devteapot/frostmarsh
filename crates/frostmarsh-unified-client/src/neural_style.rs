//! Wave 11C: Client-Side WGSL Neural Style Compute Weight Bank (`4 KB` Workgroup SRAM per Style).
//!
//! Packs 4 swappable client-side neural visual styles into `4 × 256 × vec4<f32>` (`16 KB` total,
//! exact `4,096 B` per style in WGSL `var<workgroup> w_sh: array<vec4<f32>, 256>`):
//! - Style 0: `NORDIC` — crisp boreal moss, warm weathered timber, alpine snow peaks, glacial azure
//! - Style 1: `VOLCANIC` — obsidian basalt, scorched ash-ochre, forged iron, molten lava rivers
//! - Style 2: `ARCANE` — twilight-indigo crystalline slate, bioluminescent cyan moss, plasma rivers
//! - Style 3: `EMERALD` — deep malachite-jade cavern stone, glowing emerald moss, turquoise springs

pub const NUM_Z16_STYLES: u32 = 4;
pub const STYLE_STRIDE4: usize = 256; // 256 * 16 bytes = 4,096 bytes (exact 4 KB SRAM)

const OFF4_W1: usize = 0; // 24 * 2 = 48 vec4
const OFF4_B1: usize = 48; // 6 vec4
const OFF4_W2: usize = 54; // 24 * 6 = 144 vec4
const OFF4_B2: usize = 198; // 6 vec4
const OFF4_W3: usize = 204; // 4 * 6 = 24 vec4
const OFF4_B3: usize = 228; // 1 vec4
const OFF4_SKIP: usize = 229; // 5 vec4 (4 columns + 1 bias)

pub fn z16_style_label(style_mode: u32) -> &'static str {
    match style_mode % NUM_Z16_STYLES {
        1 => "VOLCANIC",
        2 => "ARCANE",
        3 => "EMERALD",
        _ => "NORDIC",
    }
}

struct StyleWeightBuilder {
    buf: [[f32; 4]; STYLE_STRIDE4],
}

impl StyleWeightBuilder {
    fn new() -> Self {
        Self {
            buf: [[0.0; 4]; STYLE_STRIDE4],
        }
    }

    fn set_w1(&mut self, out_i: usize, w: [f32; 8], bias: f32) {
        let r = OFF4_W1 + out_i * 2;
        self.buf[r] = [w[0], w[1], w[2], w[3]];
        self.buf[r + 1] = [w[4], w[5], w[6], w[7]];
        self.buf[OFF4_B1 + out_i / 4][out_i % 4] = bias;
    }

    /// Encode a smooth C-infinity `[0, 1]` ramp detector as a pair of Layer-1 SiLU neurons
    /// `(pair_idx * 2, pair_idx * 2 + 1)` over linear combination `w · x` transitioning across `[a, b]`.
    fn set_l1_ramp_pair(&mut self, pair_idx: usize, w: [f32; 8], a: f32, b: f32) {
        let span = (b - a).max(1e-3);
        let k = 6.0 / span;
        let mut wk = [0.0f32; 8];
        for i in 0..8 {
            wk[i] = w[i] * k;
        }
        self.set_w1(pair_idx * 2, wk, -a * k);
        self.set_w1(pair_idx * 2 + 1, wk, -b * k);
    }

    fn set_w2_row(&mut self, out_i: usize, row: &[f32; 24], bias: f32) {
        let r = OFF4_W2 + out_i * 6;
        for c in 0..6 {
            self.buf[r + c] = [
                row[c * 4],
                row[c * 4 + 1],
                row[c * 4 + 2],
                row[c * 4 + 3],
            ];
        }
        self.buf[OFF4_B2 + out_i / 4][out_i % 4] = bias;
    }

    /// Layer-2 neuron computing `silu(scale * (sum_terms + offset))` where each term `(pair_idx, coeff)`
    /// reads the normalized `[0, 1]` ramp output `(h1[2*p] - h1[2*p+1]) / 6.0`.
    fn set_l2_conjunction(&mut self, out_i: usize, terms: &[(usize, f32)], offset: f32, sharpness: f32) {
        let mut row = [0.0f32; 24];
        let s = sharpness / 6.0;
        for &(p, coeff) in terms {
            row[p * 2] += coeff * s;
            row[p * 2 + 1] -= coeff * s;
        }
        self.set_w2_row(out_i, &row, offset * sharpness);
    }

    fn add_w3_col(&mut self, h2_idx: usize, rgba_delta: [f32; 4]) {
        for ch in 0..4 {
            let r = OFF4_W3 + ch * 6 + h2_idx / 4;
            self.buf[r][h2_idx % 4] += rgba_delta[ch];
        }
    }

    fn set_b3(&mut self, bias: [f32; 4]) {
        self.buf[OFF4_B3] = bias;
    }

    /// Sets the `4×4` affine skip color-grading matrix (`cols[0..4]`) + `bias` in `w_sh[229..234]`.
    fn set_skip_affine(&mut self, cols: [[f32; 4]; 4], bias: [f32; 4]) {
        for c in 0..4 {
            self.buf[OFF4_SKIP + c] = cols[c];
        }
        self.buf[OFF4_SKIP + 4] = bias;
    }

    /// Installs the shared physical feature extractor (Layer 1: 12 smooth `[0, 1]` detectors;
    /// Layer 2: 12 semantic material conjunctions).
    ///
    /// Layer 2 output slots `h2[0..12]` (each ~`0.0` when false, ~`1.0` when true, normalized by `/ 3.0`):
    /// - `0`: `solid_moss`     (solid terrain with greenish foliage/moss albedo)
    /// - `1`: `solid_wood`     (solid warm timber bridge / wood)
    /// - `2`: `solid_peak`     (solid high-altitude alpine ridge)
    /// - `3`: `solid_icy`      (solid bright icy/snowy voxel)
    /// - `4`: `solid_cave`     (solid deep/low-altitude cavern stone)
    /// - `5`: `fluid_water`    (pure fluid voxel, non-solid)
    /// - `6`: `waterfall`      (elevated plunging fluid waterfall)
    /// - `7`: `gas_plume`      (gas/smoke/steam cloud)
    /// - `8`: `gas_glow`       (hot glowing fire/ember gas)
    /// - `9`: `solid_glow`     (hot glowing solid ground/lava crust)
    /// - `10`: `solid_dark`    (dark shadow-stone/basalt solid)
    /// - `11`: `solid_mid`     (mid-altitude solid rock/cliff)
    fn install_shared_basis(&mut self) {
        // Input x = [r, g, b, glow, solid, fluid, gas, y_norm]
        // Layer 1: 12 smooth [0, 1] ramp pairs (24 neurons total)
        const P_SOLID: usize = 0;
        const P_FLUID: usize = 1;
        const P_GAS: usize = 2;
        const P_GREEN: usize = 3;
        const P_WARM: usize = 4;
        const P_ICY: usize = 5;
        const P_HIGH_Y: usize = 6;
        const P_LOW_Y: usize = 7;
        const P_MID_Y: usize = 8;
        const P_GLOW: usize = 9;
        const P_BRIGHT: usize = 10;
        const P_DARK: usize = 11;

        self.set_l1_ramp_pair(P_SOLID, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.22, 0.55);
        self.set_l1_ramp_pair(P_FLUID, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.10, 0.35);
        self.set_l1_ramp_pair(P_GAS, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.02, 0.18);
        self.set_l1_ramp_pair(P_GREEN, [-0.55, 1.0, -0.55, 0.0, 0.0, 0.0, 0.0, 0.0], 0.01, 0.13);
        self.set_l1_ramp_pair(P_WARM, [1.0, -0.15, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.06, 0.22);
        self.set_l1_ramp_pair(P_ICY, [-0.15, 0.35, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0], 0.56, 0.82);
        self.set_l1_ramp_pair(P_HIGH_Y, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.56, 0.72);
        self.set_l1_ramp_pair(P_LOW_Y, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], -0.36, -0.14);
        self.set_l1_ramp_pair(P_MID_Y, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.26, 0.48);
        self.set_l1_ramp_pair(P_GLOW, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0.04, 0.28);
        self.set_l1_ramp_pair(P_BRIGHT, [0.299, 0.587, 0.114, 0.0, 0.0, 0.0, 0.0, 0.0], 0.52, 0.80);
        self.set_l1_ramp_pair(P_DARK, [-0.299, -0.587, -0.114, 0.0, 0.0, 0.0, 0.0, 0.0], -0.40, -0.15);

        // Layer 2: 12 clean conjunction neurons. At true (sum=1.0), `silu(6.0*(1.0 - 0.5)) = silu(3.0) ≈ 2.86`.
        // We normalize W3 deltas by `1.0 / 2.86` when adding them!
        self.set_l2_conjunction(0, &[(P_SOLID, 0.65), (P_GREEN, 0.85)], -1.00, 6.0); // solid_moss
        self.set_l2_conjunction(1, &[(P_SOLID, 0.60), (P_WARM, 0.85), (P_BRIGHT, -0.45)], -0.95, 6.0); // solid_wood
        self.set_l2_conjunction(2, &[(P_SOLID, 0.65), (P_HIGH_Y, 0.85), (P_WARM, -0.50)], -1.00, 6.0); // solid_peak
        self.set_l2_conjunction(3, &[(P_SOLID, 0.65), (P_ICY, 0.85)], -1.00, 6.0); // solid_icy
        self.set_l2_conjunction(4, &[(P_SOLID, 0.65), (P_LOW_Y, 0.85)], -1.00, 6.0); // solid_cave
        self.set_l2_conjunction(5, &[(P_FLUID, 0.90), (P_SOLID, -0.55)], -0.40, 6.0); // fluid_water
        self.set_l2_conjunction(6, &[(P_FLUID, 0.70), (P_MID_Y, 0.80), (P_SOLID, -0.45)], -1.00, 6.0); // waterfall
        self.set_l2_conjunction(7, &[(P_GAS, 0.95), (P_SOLID, -0.40)], -0.45, 6.0); // gas_plume
        self.set_l2_conjunction(8, &[(P_GAS, 0.65), (P_GLOW, 0.85)], -1.00, 6.0); // gas_glow
        self.set_l2_conjunction(9, &[(P_SOLID, 0.65), (P_GLOW, 0.85)], -1.00, 6.0); // solid_glow
        self.set_l2_conjunction(10, &[(P_SOLID, 0.65), (P_DARK, 0.85)], -1.00, 6.0); // solid_dark
        self.set_l2_conjunction(11, &[(P_SOLID, 0.65), (P_MID_Y, 0.85), (P_HIGH_Y, -0.55)], -1.00, 6.0); // solid_mid
    }

    fn add_feature_delta(&mut self, h2_idx: usize, rgba: [f32; 4]) {
        const INV_SILU3: f32 = 1.0 / 2.8577;
        self.add_w3_col(
            h2_idx,
            [
                rgba[0] * INV_SILU3,
                rgba[1] * INV_SILU3,
                rgba[2] * INV_SILU3,
                rgba[3] * INV_SILU3,
            ],
        );
    }
}

fn luma_tinted_skip(base: [f32; 3], scale: [f32; 3], glow_scale: f32) -> ([[f32; 4]; 4], [f32; 4]) {
    // base_rgb + scale_rgb * luma(r, g, b)
    let wr = 0.299;
    let wg = 0.587;
    let wb = 0.114;
    let cols = [
        [scale[0] * wr, scale[1] * wr, scale[2] * wr, 0.0],
        [scale[0] * wg, scale[1] * wg, scale[2] * wg, 0.0],
        [scale[0] * wb, scale[1] * wb, scale[2] * wb, 0.0],
        [0.0, 0.0, 0.0, glow_scale],
    ];
    let bias = [base[0], base[1], base[2], 0.0];
    (cols, bias)
}

/// Builds the `4 × 256 × vec4<f32>` (`16 KB`, `4 KB` per style) client-side neural style weight bank.
pub fn build_neural_style_weights() -> Vec<[f32; 4]> {
    let mut all = Vec::with_capacity(NUM_Z16_STYLES as usize * STYLE_STRIDE4);

    // -------------------------------------------------------------------------
    // Style 0: NORDIC — preserves physical server albedo, enhances emerald boreal moss,
    // warm timber, high-peak alpine snow & foaming waterfalls!
    // -------------------------------------------------------------------------
    {
        let mut s = StyleWeightBuilder::new();
        s.install_shared_basis();
        s.set_skip_affine(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [0.0, 0.0, 0.0, 0.0],
        );
        s.add_feature_delta(0, [-0.04, 0.08, -0.02, 0.0]); // solid_moss: lush emerald boost
        s.add_feature_delta(1, [0.05, 0.02, -0.03, 0.0]); // solid_wood: warm timber
        s.add_feature_delta(2, [0.34, 0.36, 0.40, 0.0]); // solid_peak: crisp alpine snow
        s.add_feature_delta(3, [0.06, 0.07, 0.09, 0.0]); // solid_icy: frost brightness
        s.add_feature_delta(5, [-0.02, 0.04, 0.08, 0.0]); // fluid_water: glacial clarity
        s.add_feature_delta(6, [0.28, 0.30, 0.28, 0.0]); // waterfall: white foam highlight
        s.set_b3([0.0, 0.0, 0.0, 0.0]);
        all.extend_from_slice(&s.buf);
    }

    // -------------------------------------------------------------------------
    // Style 1: VOLCANIC EMBER — obsidian basalt, scorched ash-ochre, forged iron,
    // molten lava rivers & incandescent lava falls!
    // -------------------------------------------------------------------------
    {
        let mut s = StyleWeightBuilder::new();
        s.install_shared_basis();
        let (cols, bias) = luma_tinted_skip([0.11, 0.09, 0.11], [0.28, 0.22, 0.22], 1.15);
        s.set_skip_affine(cols, bias);
        s.add_feature_delta(0, [0.24, 0.09, -0.02, 0.02]); // solid_moss -> scorched ash-ochre
        s.add_feature_delta(1, [0.20, 0.08, 0.00, 0.00]); // solid_wood -> forged bronze/charred timber
        s.add_feature_delta(2, [0.22, 0.19, 0.17, 0.00]); // solid_peak -> smoky pumice ash peak
        s.add_feature_delta(3, [0.16, 0.14, 0.13, 0.00]); // solid_icy -> pale volcanic ash
        s.add_feature_delta(4, [0.14, 0.02, -0.04, 0.14]); // solid_cave -> warm glowing magma stone
        s.add_feature_delta(5, [0.78, 0.22, -0.06, 0.72]); // fluid_water -> molten lava river
        s.add_feature_delta(6, [0.12, 0.36, 0.08, 0.24]); // waterfall -> blazing lava cascade
        s.add_feature_delta(7, [0.16, 0.08, 0.06, 0.04]); // gas_plume -> volcanic soot
        s.add_feature_delta(8, [0.55, 0.28, 0.04, 0.22]); // gas_glow -> blazing ember flame
        s.add_feature_delta(9, [0.50, 0.20, 0.02, 0.20]); // solid_glow -> molten crust
        all.extend_from_slice(&s.buf);
    }

    // -------------------------------------------------------------------------
    // Style 2: ARCANE THERMAL — twilight-indigo crystalline slate, bioluminescent
    // cyan-teal moss, violet-gold runic timber, arcane plasma rivers!
    // -------------------------------------------------------------------------
    {
        let mut s = StyleWeightBuilder::new();
        s.install_shared_basis();
        let (cols, bias) = luma_tinted_skip([0.08, 0.12, 0.25], [0.18, 0.26, 0.40], 1.12);
        s.set_skip_affine(cols, bias);
        s.add_feature_delta(0, [-0.02, 0.28, 0.24, 0.06]); // solid_moss -> bioluminescent cyan-teal
        s.add_feature_delta(1, [0.34, 0.16, 0.34, 0.05]); // solid_wood -> runic violet-gold
        s.add_feature_delta(2, [0.34, 0.52, 0.46, 0.06]); // solid_peak -> luminous crystal peaks
        s.add_feature_delta(3, [0.22, 0.36, 0.32, 0.04]); // solid_icy -> arctic arcane crystal
        s.add_feature_delta(4, [0.06, 0.04, 0.18, 0.10]); // solid_cave -> twilight rune floor
        s.add_feature_delta(5, [-0.04, 0.56, 0.62, 0.56]); // fluid_water -> electric arcane plasma
        s.add_feature_delta(6, [0.40, -0.06, 0.16, 0.26]); // waterfall -> magenta-cyan energy falls
        s.add_feature_delta(7, [0.06, 0.18, 0.32, 0.05]); // gas_plume -> arcane mist
        s.add_feature_delta(8, [0.46, 0.16, 0.44, 0.20]); // gas_glow -> plasma flame
        s.add_feature_delta(9, [0.38, 0.22, 0.46, 0.16]); // solid_glow -> energized thermal rune
        all.extend_from_slice(&s.buf);
    }

    // -------------------------------------------------------------------------
    // Style 3: EMERALD CAVERN — deep malachite-jade cavern stone, glowing emerald
    // moss, gilded amber-cedar bridges, luminous turquoise mineral rivers!
    // -------------------------------------------------------------------------
    {
        let mut s = StyleWeightBuilder::new();
        s.install_shared_basis();
        let (cols, bias) = luma_tinted_skip([0.05, 0.12, 0.11], [0.14, 0.28, 0.22], 1.08);
        s.set_skip_affine(cols, bias);
        s.add_feature_delta(0, [-0.02, 0.34, 0.12, 0.10]); // solid_moss -> glowing emerald moss
        s.add_feature_delta(1, [0.40, 0.22, -0.04, 0.04]); // solid_wood -> warm amber-cedar bridge
        s.add_feature_delta(2, [0.38, 0.54, 0.44, 0.05]); // solid_peak -> pale quartz-jade peaks
        s.add_feature_delta(3, [0.24, 0.38, 0.32, 0.04]); // solid_icy -> crystal sea-jade ice
        s.add_feature_delta(4, [-0.02, 0.22, 0.14, 0.18]); // solid_cave -> bioluminescent jade floor
        s.add_feature_delta(5, [-0.02, 0.58, 0.44, 0.40]); // fluid_water -> glowing turquoise mineral river
        s.add_feature_delta(6, [0.22, 0.28, 0.24, 0.24]); // waterfall -> radiant emerald cascade
        s.add_feature_delta(7, [0.06, 0.26, 0.18, 0.04]); // gas_plume -> glowing cavern spore mist
        s.add_feature_delta(8, [0.18, 0.56, 0.26, 0.18]); // gas_glow -> emerald alchemical fire
        s.add_feature_delta(9, [0.16, 0.48, 0.22, 0.16]); // solid_glow -> jade crystal vein
        s.add_feature_delta(10, [-0.02, 0.06, 0.05, 0.03]); // solid_dark -> deep malachite depth
        all.extend_from_slice(&s.buf);
    }

    all
}

/// CPU reference evaluator matching `assets/shaders/neural_style_z16.wgsl` 1:1 (for tests & parity).
#[cfg(test)]
pub fn cpu_eval_style_voxel(
    weights: &[[f32; 4]],
    style_id: u32,
    dm: [f32; 4],
    at: [f32; 4],
    y_norm: f32,
) -> [f32; 4] {
    if dm[0] + dm[1] + dm[2] + at[0] + at[1] + at[2] + at[3] < 0.001 {
        return [0.0; 4];
    }
    let w = &weights[(style_id as usize % NUM_Z16_STYLES as usize) * STYLE_STRIDE4..];
    let x = [at[0], at[1], at[2], at[3], dm[0], dm[1], dm[2], y_norm];
    let silu = |v: f32| -> f32 {
        let c = v.clamp(-20.0, 20.0);
        c / (1.0 + (-c).exp())
    };
    let dot4 = |a: [f32; 4], b: [f32; 4]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

    let x0 = [x[0], x[1], x[2], x[3]];
    let x1 = [x[4], x[5], x[6], x[7]];
    let mut h1 = [0.0f32; 24];
    for i in 0..24 {
        let r = OFF4_W1 + i * 2;
        let b = w[OFF4_B1 + i / 4][i % 4];
        h1[i] = silu(b + dot4(w[r], x0) + dot4(w[r + 1], x1));
    }

    let mut h2 = [0.0f32; 24];
    for i in 0..24 {
        let r = OFF4_W2 + i * 6;
        let mut acc = w[OFF4_B2 + i / 4][i % 4];
        for c in 0..6 {
            acc += dot4(w[r + c], [h1[c * 4], h1[c * 4 + 1], h1[c * 4 + 2], h1[c * 4 + 3]]);
        }
        h2[i] = silu(acc);
    }

    let mut out = [0.0f32; 4];
    for ch in 0..4 {
        let r = OFF4_W3 + ch * 6;
        let mut delta = w[OFF4_B3][ch];
        for c in 0..6 {
            delta += dot4(w[r + c], [h2[c * 4], h2[c * 4 + 1], h2[c * 4 + 2], h2[c * 4 + 3]]);
        }
        let base = w[OFF4_SKIP + 4][ch]
            + w[OFF4_SKIP][ch] * x0[0]
            + w[OFF4_SKIP + 1][ch] * x0[1]
            + w[OFF4_SKIP + 2][ch] * x0[2]
            + w[OFF4_SKIP + 3][ch] * x0[3];
        out[ch] = (base + delta).clamp(0.0, 1.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_four_styles_sram_weights_distinct_and_finite() {
        let w = build_neural_style_weights();
        assert_eq!(w.len(), NUM_Z16_STYLES as usize * STYLE_STRIDE4);
        assert_eq!(STYLE_STRIDE4 * std::mem::size_of::<[f32; 4]>(), 4096);

        let moss_dm = [0.95, 0.0, 0.0, 0.8];
        let moss_at = [0.22, 0.56, 0.24, 0.0];
        let mut outs = [[0.0f32; 4]; 4];
        for sid in 0..NUM_Z16_STYLES {
            let y = cpu_eval_style_voxel(&w, sid, moss_dm, moss_at, 0.35);
            for c in y {
                assert!(c.is_finite() && (0.0..=1.0).contains(&c));
            }
            outs[sid as usize] = y;
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                let diff: f32 = (0..4).map(|k| (outs[i][k] - outs[j][k]).abs()).sum();
                assert!(diff > 0.12, "Styles {i} and {j} should diverge clearly (diff={diff})");
            }
        }
    }
}
