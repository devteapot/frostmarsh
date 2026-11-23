// Stage A: HD Pixel-Art 2x Supersampled Resolve (with Clean-3D toggle via crisp < 0.5)
// Features: 64-color hue-shifted palette harmonization, screen-space depth-crevice contact AO,
// cool-indigo silhouette & depth-edge ink outlines, and subtle Bayer 4x4 micro-dither.

struct ResolveUniforms {
    low_res: vec2<f32>,
    ss: f32,
    outline: f32,
    quantise: f32,
    dither: f32,
    crisp: f32,
    palette_count: f32,
    background: vec4<f32>,
    palette: array<vec4<f32>, 64>,
};

@group(0) @binding(0) var<uniform> u: ResolveUniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pts[vid], 0.0, 1.0);
}

const SOLID_ALPHA: f32 = 0.88;

fn load_texel(coord: vec2<i32>) -> vec4<f32> {
    let max_c = vec2<i32>(u.low_res * u.ss) - vec2<i32>(1, 1);
    return textureLoad(src_tex, clamp(coord, vec2<i32>(0, 0), max_c), 0);
}

fn load_depth(cell: vec2<i32>) -> f32 {
    let max_c = vec2<i32>(u.low_res * u.ss) - vec2<i32>(1, 1);
    let c = clamp(cell * i32(u.ss), vec2<i32>(0, 0), max_c);
    return textureLoad(depth_tex, c, 0);
}

fn coverage(cell: vec2<i32>) -> f32 {
    let lim = vec2<i32>(u.low_res);
    if (cell.x < 0 || cell.y < 0 || cell.x >= lim.x || cell.y >= lim.y) { return 1.0; }
    if (u.ss < 1.5) { return select(0.0, 1.0, load_texel(cell).a >= SOLID_ALPHA); }
    let t = cell * 2;
    return 0.25 * (
        select(0.0, 1.0, load_texel(t).a >= SOLID_ALPHA)
      + select(0.0, 1.0, load_texel(t + vec2<i32>(1, 0)).a >= SOLID_ALPHA)
      + select(0.0, 1.0, load_texel(t + vec2<i32>(0, 1)).a >= SOLID_ALPHA)
      + select(0.0, 1.0, load_texel(t + vec2<i32>(1, 1)).a >= SOLID_ALPHA)
    );
}

fn nearest_palette(col: vec3<f32>, solid: bool) -> i32 {
    let n = i32(u.palette_count);
    var best: i32 = 0;
    var best_d: f32 = 1e9;
    for (var i: i32 = 0; i < 64; i = i + 1) {
        if (i >= n) { break; }
        let p = u.palette[i].rgb;
        if (solid && n > 1 && distance(p, u.background.rgb) < 0.01) { continue; }
        let e = col - p;
        let w = e * vec3<f32>(0.30, 0.59, 0.11);
        let d = dot(w, w) + 0.28 * dot(e, e);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

fn bayer4(cell: vec2<i32>) -> f32 {
    var m = array<f32, 16>(
        0.0,  8.0,  2.0, 10.0,
        12.0, 4.0, 14.0,  6.0,
        3.0, 11.0,  1.0,  9.0,
        15.0, 7.0, 13.0,  5.0
    );
    return (m[(cell.x & 3) + ((cell.y & 3) << 2)] + 0.5) / 16.0;
}

fn ink_color(body: vec3<f32>) -> vec3<f32> {
    let tinted = body * vec3<f32>(0.34, 0.36, 0.46);
    return clamp(tinted, vec3<f32>(0.04, 0.05, 0.10), vec3<f32>(1.0));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let cell = vec2<i32>(floor(pos.xy));
    let n_samples = select(1, 4, u.ss >= 1.5);

    var sum = vec4<f32>(0.0);
    var solid_sum = vec4<f32>(0.0);

    for (var i: i32 = 0; i < 4; i = i + 1) {
        if (i >= n_samples) { break; }
        let offset = vec2<i32>(i & 1, i >> 1);
        let coord = select(cell, cell * 2 + offset, n_samples == 4);
        let s = load_texel(coord);
        sum += s;
        if (s.a >= SOLID_ALPHA) {
            solid_sum += vec4<f32>(s.rgb / max(s.a, 0.001), 1.0);
        }
    }
    sum /= f32(n_samples);
    let cov = solid_sum.a / f32(n_samples);

    var col: vec3<f32>;
    if (u.crisp < 0.5) {
        // Clean 3D mode: smooth supersampled resolve with subtle crevice contact AO
        let d0 = load_depth(cell);
        let d_min = min(
            min(load_depth(cell + vec2<i32>(-1, 0)), load_depth(cell + vec2<i32>(1, 0))),
            min(load_depth(cell + vec2<i32>(0, -1)), load_depth(cell + vec2<i32>(0, 1))),
        );
        let occ = clamp(((d0 - d_min) - 0.0005) * 85.0, 0.0, 1.0);
        let ao = 1.0 - 0.18 * occ;
        col = (sum.rgb * ao) + u.background.rgb * (1.0 - clamp(sum.a, 0.0, 1.0));
    } else if (cov >= 0.5) {
        let continuous_col = solid_sum.rgb / max(solid_sum.a, 1.0);
        let d0 = load_depth(cell);
        let dl = load_depth(cell + vec2<i32>(-1, 0));
        let dr = load_depth(cell + vec2<i32>( 1, 0));
        let du = load_depth(cell + vec2<i32>( 0,-1));
        let dd = load_depth(cell + vec2<i32>( 0, 1));
        let d_max = max(max(dl, dr), max(du, dd));
        let d_min = min(min(dl, dr), min(du, dd));

        let occ = clamp(((d0 - d_min) - 0.0005) * 95.0, 0.0, 1.0);
        let ao = 1.0 - 0.24 * occ;

        var is_edge = false;
        if (u.outline > 0.5) {
            let n_solid = i32(coverage(cell + vec2<i32>( 1,  0)) >= 0.5)
                        + i32(coverage(cell + vec2<i32>(-1,  0)) >= 0.5)
                        + i32(coverage(cell + vec2<i32>( 0,  1)) >= 0.5)
                        + i32(coverage(cell + vec2<i32>( 0, -1)) >= 0.5);
            is_edge = (n_solid > 0 && n_solid < 4) || (d_max - d0 > 0.0028);
        }

        let shaded = continuous_col * ao;
        if (u.quantise > 0.5) {
            let d_amt = max(u.dither, 0.16);
            let d_shift = (bayer4(cell) - 0.5) * 0.075 * d_amt;
            let pal_col = u.palette[nearest_palette(clamp(shaded + vec3<f32>(d_shift), vec3<f32>(0.0), vec3<f32>(1.0)), true)].rgb;
            col = mix(shaded, pal_col, 0.68);
        } else {
            col = shaded;
        }

        if (is_edge) {
            col = ink_color(col);
        }
    } else {
        col = u.background.rgb;
    }

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
