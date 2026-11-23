//! Curated 64-color Approach 4 Frostmarsh Palette + 1 KB Animated `flash()` Palette Table.
//! Cool-indigo shadow pull (#261e48) -> warm-cream highlight push (#fff4d6).

#[derive(Clone, Copy)]
pub struct PalEntry {
    pub hex: u32,
    pub glow: f32,
}

pub const PALETTE_64: &[PalEntry] = &[
    PalEntry { hex: 0x000000, glow: 0.0 },  // 0: empty
    PalEntry { hex: 0x161430, glow: 0.0 },  // 1: deep indigo ink
    PalEntry { hex: 0x261e48, glow: 0.0 },  // 2: cool-indigo shadow pull (#261e48)
    PalEntry { hex: 0x2f3b5c, glow: 0.0 },  // 3: dark granite
    PalEntry { hex: 0x48587a, glow: 0.0 },  // 4: slate stone
    PalEntry { hex: 0x65799e, glow: 0.0 },  // 5: mid stone
    PalEntry { hex: 0x89a0c0, glow: 0.0 },  // 6: light granite
    PalEntry { hex: 0xb6c9e0, glow: 0.0 },  // 7: frost stone
    PalEntry { hex: 0xe6eff8, glow: 0.0 },  // 8: snow highlight
    PalEntry { hex: 0x112a26, glow: 0.0 },  // 9: deep pine ink
    PalEntry { hex: 0x1c4435, glow: 0.0 },  // 10: forest shadow
    PalEntry { hex: 0x2c6942, glow: 0.0 },  // 11: pine mid
    PalEntry { hex: 0x428d4d, glow: 0.0 },  // 12: marsh grass
    PalEntry { hex: 0x6bb453, glow: 0.0 },  // 13: sunlit meadow
    PalEntry { hex: 0x9cd464, glow: 0.0 },  // 14: bright canopy
    PalEntry { hex: 0xd0eb86, glow: 0.0 },  // 15: sun-kissed leaf tip
    PalEntry { hex: 0x2a1a1c, glow: 0.0 },  // 16: dark char / oil ink
    PalEntry { hex: 0x452b25, glow: 0.0 },  // 17: dark bark / peat
    PalEntry { hex: 0x6b412f, glow: 0.0 },  // 18: warm soil / timber
    PalEntry { hex: 0x98613f, glow: 0.0 },  // 19: marsh loam
    PalEntry { hex: 0xc48b58, glow: 0.0 },  // 20: sandy riverbank
    PalEntry { hex: 0xe2be8f, glow: 0.0 },  // 21: warm skin / sandstone
    PalEntry { hex: 0xfff4d6, glow: 0.0 },  // 22: warm-cream highlight (#fff4d6)
    PalEntry { hex: 0x4b1222, glow: 0.0 },  // 23: crimson shadow
    PalEntry { hex: 0x821e30, glow: 0.0 },  // 24: pyro crimson dark
    PalEntry { hex: 0xba3138, glow: 0.0 },  // 25: pyro crimson banner
    PalEntry { hex: 0xe55242, glow: 0.35 }, // 26: bright flame red
    PalEntry { hex: 0xf47e38, glow: 0.72 }, // 27: wildfire orange (animated flash!)
    PalEntry { hex: 0xf9b536, glow: 0.86 }, // 28: molten amber (animated flash!)
    PalEntry { hex: 0xfff078, glow: 0.98 }, // 29: wildfire core (animated flash!)
    PalEntry { hex: 0xfffbe0, glow: 1.00 }, // 30: incandescent white-gold
    PalEntry { hex: 0x10355d, glow: 0.0 },  // 31: deep river blue (animated flash!)
    PalEntry { hex: 0x1d5f94, glow: 0.0 },  // 32: river mid turquoise (animated flash!)
    PalEntry { hex: 0x308ac4, glow: 0.0 },  // 33: sunlit river azure
    PalEntry { hex: 0x5cc4ec, glow: 0.65 }, // 34: frost ice / river crest (animated flash!)
    PalEntry { hex: 0xa8f2ff, glow: 0.95 }, // 35: glacier core / frost pulse (animated flash!)
    PalEntry { hex: 0x2b193d, glow: 0.0 },  // 36: oil slick violet shadow
    PalEntry { hex: 0x4e2d6e, glow: 0.0 },  // 37: oil iridescent purple
    PalEntry { hex: 0x7e4c9e, glow: 0.25 }, // 38: oil / arcane sheen
    PalEntry { hex: 0x1f2028, glow: 0.0 },  // 39: wrought iron / soot
    PalEntry { hex: 0x403e4a, glow: 0.0 },  // 40: ash charcoal
    PalEntry { hex: 0xa17528, glow: 0.0 },  // 41: dark bronze
    PalEntry { hex: 0xe6b942, glow: 0.25 }, // 42: golden relic gold
    PalEntry { hex: 0x5c241e, glow: 0.0 },  // 43: scorched earth dark
    PalEntry { hex: 0x893928, glow: 0.0 },  // 44: warm terracotta
    PalEntry { hex: 0xb65438, glow: 0.0 },  // 45: ember crust
    PalEntry { hex: 0xd98050, glow: 0.0 },  // 46: sunlit clay
    PalEntry { hex: 0xecac76, glow: 0.0 },  // 47: pale fawn (deer coat)
    PalEntry { hex: 0x698eb2, glow: 0.0 },  // 48: frost snow shadow
    PalEntry { hex: 0x9bc0dc, glow: 0.0 },  // 49: frost snow mid
    PalEntry { hex: 0xcfe6f5, glow: 0.0 },  // 50: frost snow light
    PalEntry { hex: 0xf2faff, glow: 0.0 },  // 51: pure snow glint
    PalEntry { hex: 0x1d3e55, glow: 0.0 },  // 52: frost pine dark
    PalEntry { hex: 0x396a84, glow: 0.0 },  // 53: frost pine mid
    PalEntry { hex: 0x67a2ba, glow: 0.0 },  // 54: frost pine hi
    PalEntry { hex: 0x4c5e2c, glow: 0.0 },  // 55: marsh moss dark
    PalEntry { hex: 0x7a9442, glow: 0.0 },  // 56: marsh moss light
    PalEntry { hex: 0xc47d2d, glow: 0.0 },  // 57: autumn reed
    PalEntry { hex: 0x22608a, glow: 0.0 },  // 58: frostbot royal blue dark
    PalEntry { hex: 0x348be0, glow: 0.0 },  // 59: frostbot cobalt tabard
    PalEntry { hex: 0x6ec3ff, glow: 0.35 }, // 60: frostbot highlight cyan
    PalEntry { hex: 0x6d615a, glow: 0.0 },  // 61: ash grey
    PalEntry { hex: 0x9b8d82, glow: 0.0 },  // 62: pale ash / mud
    PalEntry { hex: 0x5c728a, glow: 0.0 },  // 63: steel blue
];

#[inline]
pub fn hex_to_rgb(hex: u32) -> [f32; 3] {
    [
        ((hex >> 16) & 0xff) as f32 / 255.0,
        ((hex >> 8) & 0xff) as f32 / 255.0,
        (hex & 0xff) as f32 / 255.0,
    ]
}

pub fn palette_resolve_64() -> ([[f32; 4]; 64], f32) {
    let mut out = [[0.0; 4]; 64];
    let mut count = 0;
    for e in PALETTE_64.iter().skip(1) {
        if count >= 64 {
            break;
        }
        let rgb = hex_to_rgb(e.hex);
        out[count] = [rgb[0], rgb[1], rgb[2], 1.0];
        count += 1;
    }
    (out, count as f32)
}

/// 1 KB (256 x 4 bytes) animated `flash()` palette table updated per frame!
/// Animates wildfire flames (27, 28, 29), flowing river shimmer (31, 32, 34), and glacial crystal pulse (35).
pub fn palette_tex_bytes_at(sim_time: f32) -> [u8; 256 * 4] {
    let mut out = [0u8; 256 * 4];
    for (i, e) in PALETTE_64.iter().enumerate() {
        let mut rgb = hex_to_rgb(e.hex);
        let mut glow = e.glow;
        match i {
            27 => {
                // Flickering wildfire orange
                let p = 0.5 + 0.5 * (sim_time * 9.5).sin();
                rgb[0] = (rgb[0] * (0.94 + 0.10 * p)).min(1.0);
                rgb[1] = (rgb[1] * (0.78 + 0.28 * p)).min(1.0);
                glow = (0.62 + 0.26 * p).clamp(0.0, 1.0);
            }
            28 => {
                // Molten ember pulse
                let p = 0.5 + 0.5 * (sim_time * 7.2 + 1.1).sin();
                rgb[1] = (rgb[1] * (0.86 + 0.20 * p)).min(1.0);
                glow = (0.74 + 0.22 * p).clamp(0.0, 1.0);
            }
            29 => {
                // Incandescent wildfire core flicker
                let p = 0.5 + 0.5 * (sim_time * 13.0 + 2.3).sin();
                rgb[2] = (rgb[2] * (0.75 + 0.45 * p)).min(1.0);
                glow = (0.86 + 0.14 * p).clamp(0.0, 1.0);
            }
            31 | 32 => {
                // Subtle flowing river color breath
                let p = 0.5 + 0.5 * (sim_time * 2.4 + i as f32).sin();
                rgb[1] = (rgb[1] * (0.94 + 0.10 * p)).min(1.0);
                rgb[2] = (rgb[2] * (0.95 + 0.08 * p)).min(1.0);
            }
            34 | 35 => {
                // Frost / glacial crystal shimmer pulse
                let p = 0.5 + 0.5 * (sim_time * 3.6 + i as f32 * 0.7).sin();
                glow = (e.glow * (0.82 + 0.18 * p)).clamp(0.0, 1.0);
            }
            _ => {}
        }
        out[i * 4] = (rgb[0] * 255.0).round().clamp(0.0, 255.0) as u8;
        out[i * 4 + 1] = (rgb[1] * 255.0).round().clamp(0.0, 255.0) as u8;
        out[i * 4 + 2] = (rgb[2] * 255.0).round().clamp(0.0, 255.0) as u8;
        out[i * 4 + 3] = (glow * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    out
}
