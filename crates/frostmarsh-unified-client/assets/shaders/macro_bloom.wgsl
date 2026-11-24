// Wave 38A + Wave 39A: Quarter-Resolution Multi-Pass Separable HDR Bloom (`macro_bloom.wgsl`)
//
// 3 lightweight fullscreen passes at (w/4).max(1) x (h/4).max(1) Rgba16Float:
//   1. vs_bloom_fullscreen + fs_bloom_prefilter -> 4-tap box downsample + Karis-weighted soft-knee HDR extraction
//   2. vs_bloom_fullscreen + fs_bloom_blur_h    -> 9-tap horizontal separable Gaussian blur (bloom_a -> bloom_b)
//   3. vs_bloom_fullscreen + fs_bloom_blur_v    -> 9-tap vertical separable Gaussian blur   (bloom_b -> bloom_a)

struct BloomUniforms {
    texel_size: vec4<f32>, // xy = 1.0 / src_res, z = threshold (1.28), w = knee (0.42)
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var<uniform> bloom_u: BloomUniforms;

struct BloomVertOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_bloom_fullscreen(@builtin(vertex_index) vi: u32) -> BloomVertOut {
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let p = pts[vi];
    var out: BloomVertOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn soft_knee_extract(c_in: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let c = max(c_in, vec3<f32>(0.0));
    let br = max(c.r, max(c.g, c.b));
    let soft = clamp(br - threshold + knee, 0.0, 2.0 * knee);
    let soft_contrib = (soft * soft) / max(4.0 * knee, 1e-5);
    let contrib = min(max(br - threshold, soft_contrib), 6.0);
    return c * (contrib / max(br, 1e-5));
}

fn karis_weight(c: vec3<f32>) -> f32 {
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    return 1.0 / (1.0 + luma * 0.5);
}

@fragment
fn fs_bloom_prefilter(in: BloomVertOut) -> @location(0) vec4<f32> {
    let tx = bloom_u.texel_size.xy;
    let threshold = bloom_u.texel_size.z;
    let knee = bloom_u.texel_size.w;

    // 4-tap bilinear box downsample covering the full 4x4 source pixel footprint cleanly
    let c0 = soft_knee_extract(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-tx.x, -tx.y), 0.0).rgb, threshold, knee);
    let c1 = soft_knee_extract(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( tx.x, -tx.y), 0.0).rgb, threshold, knee);
    let c2 = soft_knee_extract(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-tx.x,  tx.y), 0.0).rgb, threshold, knee);
    let c3 = soft_knee_extract(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( tx.x,  tx.y), 0.0).rgb, threshold, knee);

    let w0 = karis_weight(c0);
    let w1 = karis_weight(c1);
    let w2 = karis_weight(c2);
    let w3 = karis_weight(c3);
    let sum_w = max(w0 + w1 + w2 + w3, 1e-5);
    let col = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) / sum_w;
    return vec4<f32>(col, 1.0);
}

fn blur_9tap(uv: vec2<f32>, step_uv: vec2<f32>) -> vec3<f32> {
    var acc = textureSampleLevel(src_tex, src_samp, uv, 0.0).rgb * 0.2270270270;
    acc += (textureSampleLevel(src_tex, src_samp, uv + step_uv * 1.0, 0.0).rgb
          + textureSampleLevel(src_tex, src_samp, uv - step_uv * 1.0, 0.0).rgb) * 0.1945945946;
    acc += (textureSampleLevel(src_tex, src_samp, uv + step_uv * 2.0, 0.0).rgb
          + textureSampleLevel(src_tex, src_samp, uv - step_uv * 2.0, 0.0).rgb) * 0.1216216216;
    acc += (textureSampleLevel(src_tex, src_samp, uv + step_uv * 3.0, 0.0).rgb
          + textureSampleLevel(src_tex, src_samp, uv - step_uv * 3.0, 0.0).rgb) * 0.0540540541;
    acc += (textureSampleLevel(src_tex, src_samp, uv + step_uv * 4.0, 0.0).rgb
          + textureSampleLevel(src_tex, src_samp, uv - step_uv * 4.0, 0.0).rgb) * 0.0162162162;
    return acc;
}

@fragment
fn fs_bloom_blur_h(in: BloomVertOut) -> @location(0) vec4<f32> {
    let step_x = vec2<f32>(bloom_u.texel_size.x * 1.15, 0.0);
    return vec4<f32>(blur_9tap(in.uv, step_x), 1.0);
}

@fragment
fn fs_bloom_blur_v(in: BloomVertOut) -> @location(0) vec4<f32> {
    let step_y = vec2<f32>(0.0, bloom_u.texel_size.y * 1.15);
    return vec4<f32>(blur_9tap(in.uv, step_y), 1.0);
}
