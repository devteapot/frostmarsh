// Pass 3: Stylized Pixel-2xSS <-> Clean-3D Resolve Shader (T-key live toggle)
// - Clean-3D (params.x < 0.5): full-resolution crisp pass-through with subtle crevice contact AO.
// - Pixel-2xSS (params.x >= 0.5): 2x2 supersampled art-cell box resolve with depth-edge cool-indigo
//   ink outlines, Bayer 4x4 micro-dithering, and 9-step Cool-Indigo / Warm-Cream hue-shifted ramp!

struct PostUniforms {
    viewport: vec4<f32>, // xy = width, height; zw = 1/w, 1/h
    params: vec4<f32>,   // x = pixel_mode (0=Clean-3D, 1=Pixel-2xSS), y = pixel_scale (2.0)
};

@group(0) @binding(0) var<uniform> u: PostUniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn load_src(coord: vec2<i32>) -> vec3<f32> {
    let max_c = vec2<i32>(i32(u.viewport.x) - 1, i32(u.viewport.y) - 1);
    return textureLoad(src_tex, clamp(coord, vec2<i32>(0), max_c), 0).rgb;
}

fn load_depth(coord: vec2<i32>) -> f32 {
    let max_c = vec2<i32>(i32(u.viewport.x) - 1, i32(u.viewport.y) - 1);
    return textureLoad(depth_tex, clamp(coord, vec2<i32>(0), max_c), 0);
}

fn bayer4(cell: vec2<i32>) -> f32 {
    var m = array<f32, 16>(
        0.0,  8.0,  2.0, 10.0,
        12.0, 4.0, 14.0,  6.0,
        3.0, 11.0,  1.0,  9.0,
        15.0, 7.0, 13.0,  5.0
    );
    return (m[(cell.x & 3) + ((cell.y & 3) << 2u)] + 0.5) / 16.0;
}

fn ink_color(body: vec3<f32>) -> vec3<f32> {
    return clamp(body * vec3<f32>(0.32, 0.35, 0.48), vec3<f32>(0.04, 0.05, 0.11), vec3<f32>(1.0));
}

fn hue_shift_quantize(col: vec3<f32>, cell: vec2<i32>) -> vec3<f32> {
    // 9-step Cool-Indigo shadow -> Warm-Cream highlight quantization with Bayer 4x4 micro-dither
    let dither = (bayer4(cell) - 0.5) * (0.55 / 9.0);
    let luma = dot(col, vec3<f32>(0.299, 0.587, 0.114));
    let q_luma = clamp(floor((luma + dither) * 9.0 + 0.5) / 9.0, 0.0, 1.0);
    let scale = q_luma / max(luma, 0.008);
    var q_col = clamp(col * mix(1.0, scale, 0.72), vec3<f32>(0.0), vec3<f32>(1.0));

    // Subtle cool-indigo shadows and warm-cream highlights (Frostmarsh signature ramp)
    let shadow_w = (1.0 - smoothstep(0.0, 0.42, q_luma)) * 0.14;
    let hi_w = smoothstep(0.62, 1.0, q_luma) * 0.10;
    q_col = mix(q_col, vec3<f32>(0.10, 0.14, 0.30), shadow_w);
    q_col = mix(q_col, vec3<f32>(1.00, 0.95, 0.84), hi_w);
    return clamp(q_col, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(i32(pos.x), i32(pos.y));

    if (u.params.x < 0.5) {
        // Clean-3D native full-resolution pass-through
        let c = load_src(pix);
        return vec4<f32>(c, 1.0);
    }

    // Pixel-2xSS mode: 2x2 supersampled art cell
    let ps = max(i32(u.params.y), 2);
    let cell = pix / ps;
    let base = cell * ps;

    // 2x2 supersampled color average
    let c00 = load_src(base);
    let c10 = load_src(base + vec2<i32>(1, 0));
    let c01 = load_src(base + vec2<i32>(0, 1));
    let c11 = load_src(base + vec2<i32>(1, 1));
    var col = (c00 + c10 + c01 + c11) * 0.25;

    // Depth-edge silhouette ink & crevice contact AO in art-cell space
    let d0 = load_depth(base);
    let dl = load_depth(base + vec2<i32>(-ps, 0));
    let dr = load_depth(base + vec2<i32>( ps, 0));
    let du = load_depth(base + vec2<i32>(0, -ps));
    let dd = load_depth(base + vec2<i32>(0,  ps));
    let d_min = min(min(dl, dr), min(du, dd));
    let d_max = max(max(dl, dr), max(du, dd));

    if (d0 < 0.99995) {
        let occ = clamp(((d0 - d_min) - 0.00045) * 85.0, 0.0, 1.0);
        let ao = 1.0 - 0.22 * occ;
        col *= ao;

        col = hue_shift_quantize(col, cell);

        // Crisp cool-indigo outline on strong depth discontinuities
        let edge_delta = d_max - d0;
        let is_sky_edge = d_max >= 0.9999;
        if (is_sky_edge || edge_delta > 0.0036) {
            col = mix(col, ink_color(col), 0.68);
        }
    } else {
        // Gentle Bayer-dithered sky gradient in Pixel-2xSS mode
        let dither = (bayer4(cell) - 0.5) * 0.025;
        col = clamp(floor((col + vec3<f32>(dither)) * 20.0 + 0.5) / 20.0, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // Subtle filmic vibrance & gentle edge vignette
    let uv = pos.xy * u.viewport.zw;
    let luma = dot(col, vec3<f32>(0.299, 0.587, 0.114));
    col = mix(vec3<f32>(luma), col, 1.10);
    let v_d = length((uv - vec2<f32>(0.5)) * vec2<f32>(1.20, 1.05));
    let vig = 1.0 - smoothstep(0.58, 1.15, v_d) * 0.12;
    col *= vig;

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
