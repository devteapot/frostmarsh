// Iteration 9A: Stage A 2xSS HD Pixel-Art Resolve Pass (with T-Key Clean-3D Toggle!)
// Adapted from Frostmarsh's signature Approach 4 Stage A resolve:
//   - 2x2 Supersampled cell box-filter into crisp HD pixel-art cells
//   - Linearized perspective depth-edge ink outlines (uniform crisp silhouettes close & far!)
//   - Screen-space depth-crevice contact AO
//   - 9-step Cool-Indigo (#261e48) / Warm-Cream (#fff4d6) hue-shifted 64-color palette harmonization
//   - Subtle Bayer 4x4 micro-dither + filmic contrast + atmospheric vignette
//   - Live T-key toggle (`crisp < 0.5` -> smooth native Clean-3D with subtle contact AO!)

struct ResolveUniforms {
    out_res: vec2<f32>,      // output texture dimensions (W, H)
    cell_scale: f32,         // 2.0 for 2xSS HD pixel art (960x540 cells at 1080p)
    outline: f32,            // 1.0 = enabled
    quantise: f32,           // 1.0 = 64-color palette harmonization
    dither: f32,             // Bayer 4x4 strength (~0.24)
    crisp: f32,              // 1.0 = PIXEL-2XSS, 0.0 = CLEAN-3D
    palette_count: f32,
    depth_params: vec4<f32>, // x = near (0.25), y = far (420.0), z = style_id, w = unused
    ink_tint: vec4<f32>,     // rgb = cool-indigo / obsidian / deep-violet ink tint, w = mix weight
    palette: array<vec4<f32>, 64>,
};

@group(0) @binding(0) var<uniform> u: ResolveUniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    return vec4<f32>(pts[vid], 0.0, 1.0);
}

fn load_src(coord: vec2<i32>) -> vec4<f32> {
    let max_c = vec2<i32>(u.out_res) - vec2<i32>(1, 1);
    return textureLoad(src_tex, clamp(coord, vec2<i32>(0, 0), max_c), 0);
}

fn linearize_depth(d: f32) -> f32 {
    let near = u.depth_params.x;
    let far = u.depth_params.y;
    if (d >= 0.99998) {
        return far;
    }
    return (near * far) / max(far - d * (far - near), 1e-4);
}

fn load_lin_depth_cell(cell: vec2<i32>, scale: i32) -> f32 {
    let max_c = vec2<i32>(u.out_res) - vec2<i32>(1, 1);
    let c = clamp(cell * scale, vec2<i32>(0, 0), max_c);
    return linearize_depth(textureLoad(depth_tex, c, 0));
}

fn bayer4(cell: vec2<i32>) -> f32 {
    var m = array<f32, 16>(
        0.0,  8.0,  2.0, 10.0,
        12.0, 4.0, 14.0,  6.0,
        3.0, 11.0,  1.0,  9.0,
        15.0, 7.0, 13.0,  5.0,
    );
    return (m[(cell.x & 3) + ((cell.y & 3) << 2)] + 0.5) / 16.0;
}

fn nearest_palette(col: vec3<f32>) -> vec3<f32> {
    let n = i32(u.palette_count);
    var best_col = col;
    var best_d = 1e9;
    for (var i: i32 = 0; i < 64; i = i + 1) {
        if (i >= n) {
            break;
        }
        let p = u.palette[i].rgb;
        let e = col - p;
        let w = e * vec3<f32>(0.30, 0.59, 0.11);
        let d = dot(w, w) + 0.28 * dot(e, e);
        if (d < best_d) {
            best_d = d;
            best_col = p;
        }
    }
    return best_col;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(floor(pos.xy));
    var col: vec3<f32>;

    if (u.crisp < 0.5) {
        // CLEAN-3D mode: native smooth supersampled output + subtle contact depth AO
        let s = load_src(pix);
        let z0 = linearize_depth(textureLoad(depth_tex, pix, 0));
        let max_c = vec2<i32>(u.out_res) - vec2<i32>(1, 1);
        let zl = linearize_depth(textureLoad(depth_tex, clamp(pix + vec2<i32>(-1, 0), vec2<i32>(0), max_c), 0));
        let zr = linearize_depth(textureLoad(depth_tex, clamp(pix + vec2<i32>( 1, 0), vec2<i32>(0), max_c), 0));
        let zu = linearize_depth(textureLoad(depth_tex, clamp(pix + vec2<i32>( 0,-1), vec2<i32>(0), max_c), 0));
        let zd = linearize_depth(textureLoad(depth_tex, clamp(pix + vec2<i32>( 0, 1), vec2<i32>(0), max_c), 0));
        let z_min = min(min(zl, zr), min(zu, zd));
        let occ = clamp(((z0 - z_min) / max(z0, 1.0) - 0.015) * 4.5, 0.0, 1.0) * select(0.0, 1.0, z0 < 350.0);
        col = s.rgb * (1.0 - 0.18 * occ);
    } else {
        // PIXEL-2XSS mode: 2x2 supersampled resolve into crisp HD pixel-art cells!
        let sc = max(i32(round(u.cell_scale)), 1);
        let cell = pix / sc;

        var sum = vec3<f32>(0.0);
        var geo_cov = 0.0;
        let base = cell * sc;
        if (sc >= 2) {
            let s00 = load_src(base + vec2<i32>(0, 0));
            let s10 = load_src(base + vec2<i32>(1, 0));
            let s01 = load_src(base + vec2<i32>(0, 1));
            let s11 = load_src(base + vec2<i32>(1, 1));
            sum = (s00.rgb + s10.rgb + s01.rgb + s11.rgb) * 0.25;
            geo_cov = (s00.a + s10.a + s01.a + s11.a) * 0.25;
        } else {
            let s00 = load_src(base);
            sum = s00.rgb;
            geo_cov = s00.a;
        }

        let z0 = load_lin_depth_cell(cell, sc);
        let zl = load_lin_depth_cell(cell + vec2<i32>(-1, 0), sc);
        let zr = load_lin_depth_cell(cell + vec2<i32>( 1, 0), sc);
        let zu = load_lin_depth_cell(cell + vec2<i32>( 0,-1), sc);
        let zd = load_lin_depth_cell(cell + vec2<i32>( 0, 1), sc);
        let z_max = max(max(zl, zr), max(zu, zd));
        let z_min = min(min(zl, zr), min(zu, zd));

        let is_solid = z0 < 360.0;
        // Crevice contact AO where a pixel sits recessed behind an adjacent foreground surface
        let occ = clamp(((z0 - z_min) / max(z0, 1.0) - 0.018) * 5.0, 0.0, 1.0) * select(0.0, 1.0, is_solid);
        let ao = 1.0 - 0.22 * occ;

        // Crisp depth-edge & silhouette ink outline (foreground edge only -> clean 1-cell ink line!)
        var is_edge = false;
        if (u.outline > 0.5 && is_solid) {
            let rel_jump = (z_max - z0) / max(z0, 2.0);
            let sky_sil = z_max > 360.0 && z0 < 240.0;
            is_edge = sky_sil || (rel_jump > 0.13);
        }

        var shaded = sum * ao;

        // Gentle Bayer 4x4 micro-dither + 64-color style palette harmonization!
        if (u.quantise > 0.5 && u.palette_count > 1.0) {
            let d_shift = (bayer4(cell) - 0.5) * 0.055 * max(u.dither, 0.16);
            let q_in = clamp(shaded + vec3<f32>(d_shift), vec3<f32>(0.0), vec3<f32>(1.0));
            let pal_col = nearest_palette(q_in);
            // Keep rich sky gradients smooth while giving solid terrain/architecture/pines crisp palette cohesion
            let pal_mix = mix(0.14, 0.38, clamp(geo_cov, 0.0, 1.0));
            shaded = mix(shaded, pal_col, pal_mix);
        }

        if (is_edge) {
            let ink = clamp(shaded * u.ink_tint.rgb, vec3<f32>(0.03, 0.03, 0.08), vec3<f32>(1.0));
            shaded = mix(shaded, ink, u.ink_tint.w);
        }

        col = shaded;
    }

    // Subtle filmic S-curve + vibrance + gentle atmospheric edge vignette
    let uv = pos.xy / max(u.out_res, vec2<f32>(1.0));
    let luma = dot(col, vec3<f32>(0.299, 0.587, 0.114));
    col = mix(vec3<f32>(luma), col, 1.12);
    col = col * col * (3.0 - 2.0 * col) * 0.14 + col * 0.86;
    let v_d = length((uv - vec2<f32>(0.5)) * vec2<f32>(1.22, 1.04));
    let vig = 1.0 - smoothstep(0.58, 1.18, v_d) * 0.13;
    col *= vig;

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
