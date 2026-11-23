// Wave 11C: Live Client-Side WGSL Neural Style Compute Engine (`Z=16..32`)
// Loads 4 KB (`256 × vec4<f32>`) per-style neural weights into ultra-fast workgroup SRAM
// (1 coalesced vec4 load per thread across `@workgroup_size(8, 4, 8)` = 256 threads!),
// evaluates an unrolled `8 -> 24 -> 24 -> 4` SiLU Residual Style MLP over raw physical
// `(density_mat, albedo_thermal, y_norm)` voxels, and writes styled RGBA8 directly to
// `z16_styled_albedo_tex` (`texture_storage_3d<rgba8unorm, write>`)!
//
// Supports arbitrary depth `D` (`16..32`), incremental dirty `16×D×16` brick dispatches,
// and instantaneous full-volume `Y`-key style swaps across 4 styles:
//   0 = NORDIC, 1 = VOLCANIC EMBER, 2 = ARCANE THERMAL, 3 = EMERALD CAVERN.

struct StyleComputeParams {
    // x = width, y = depth (16..32), z = height, w = style_id (0..3)
    vol_dims: vec4<u32>,
    region_min: vec4<u32>,
    region_size: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: StyleComputeParams;
@group(0) @binding(1) var density_in: texture_3d<f32>;
@group(0) @binding(2) var albedo_in: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(4) var styled_albedo_out: texture_storage_3d<rgba8unorm, write>;

// Exact 4 KB (`256 vec4<f32>` = `1,024 f32`) per style in workgroup SRAM:
//   W1:       24 * 2 vec4 =  48 vec4  [  0.. 48)
//   b1:        6     vec4 =   6 vec4  [ 48.. 54)
//   W2:       24 * 6 vec4 = 144 vec4  [ 54..198)
//   b2:        6     vec4 =   6 vec4  [198..204)
//   W3:        4 * 6 vec4 =  24 vec4  [204..228)
//   b3:        1     vec4 =   1 vec4  [228..229)
//   SkipBase:  5     vec4 =   5 vec4  [229..234)  (4 affine cols + 1 bias)
//   Meta/Pad: 22     vec4 =  22 vec4  [234..256)
const STYLE_STRIDE4: u32 = 256u;
const OFF4_W1: u32 = 0u;
const OFF4_B1: u32 = 48u;
const OFF4_W2: u32 = 54u;
const OFF4_B2: u32 = 198u;
const OFF4_W3: u32 = 204u;
const OFF4_B3: u32 = 228u;
const OFF4_SKIP: u32 = 229u;

var<workgroup> w_sh: array<vec4<f32>, 256>;

fn silu4(v: vec4<f32>) -> vec4<f32> {
    let clamped = clamp(v, vec4<f32>(-20.0), vec4<f32>(20.0));
    return clamped / (vec4<f32>(1.0) + exp(-clamped));
}

fn dot24(row4: u32, h0: vec4<f32>, h1: vec4<f32>, h2: vec4<f32>, h3: vec4<f32>, h4: vec4<f32>, h5: vec4<f32>) -> f32 {
    return dot(w_sh[row4 + 0u], h0)
         + dot(w_sh[row4 + 1u], h1)
         + dot(w_sh[row4 + 2u], h2)
         + dot(w_sh[row4 + 3u], h3)
         + dot(w_sh[row4 + 4u], h4)
         + dot(w_sh[row4 + 5u], h5);
}

fn l1_block(g: u32, x0: vec4<f32>, x1: vec4<f32>) -> vec4<f32> {
    let r = OFF4_W1 + g * 8u;
    let b = w_sh[OFF4_B1 + g];
    return silu4(b + vec4<f32>(
        dot(w_sh[r + 0u], x0) + dot(w_sh[r + 1u], x1),
        dot(w_sh[r + 2u], x0) + dot(w_sh[r + 3u], x1),
        dot(w_sh[r + 4u], x0) + dot(w_sh[r + 5u], x1),
        dot(w_sh[r + 6u], x0) + dot(w_sh[r + 7u], x1),
    ));
}

fn l2_block(g: u32, h0: vec4<f32>, h1: vec4<f32>, h2: vec4<f32>, h3: vec4<f32>, h4: vec4<f32>, h5: vec4<f32>) -> vec4<f32> {
    let r = OFF4_W2 + g * 24u;
    let b = w_sh[OFF4_B2 + g];
    return silu4(b + vec4<f32>(
        dot24(r +  0u, h0, h1, h2, h3, h4, h5),
        dot24(r +  6u, h0, h1, h2, h3, h4, h5),
        dot24(r + 12u, h0, h1, h2, h3, h4, h5),
        dot24(r + 18u, h0, h1, h2, h3, h4, h5),
    ));
}

fn l3_head(h0: vec4<f32>, h1: vec4<f32>, h2: vec4<f32>, h3: vec4<f32>, h4: vec4<f32>, h5: vec4<f32>) -> vec4<f32> {
    let r = OFF4_W3;
    let b = w_sh[OFF4_B3];
    return b + vec4<f32>(
        dot24(r +  0u, h0, h1, h2, h3, h4, h5),
        dot24(r +  6u, h0, h1, h2, h3, h4, h5),
        dot24(r + 12u, h0, h1, h2, h3, h4, h5),
        dot24(r + 18u, h0, h1, h2, h3, h4, h5),
    );
}

@compute @workgroup_size(8, 4, 8)
fn cs_style_z16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // 1 coalesced vec4<f32> load per thread -> 4 KB workgroup SRAM!
    w_sh[lid] = weights[params.vol_dims.w * STYLE_STRIDE4 + lid];
    workgroupBarrier();

    if (gid.x >= params.region_size.x || gid.y >= params.region_size.y || gid.z >= params.region_size.z) {
        return;
    }

    let x = params.region_min.x + gid.x;
    let y = params.region_min.y + gid.y;
    let z = params.region_min.z + gid.z;

    let w = params.vol_dims.x;
    let d = params.vol_dims.y;
    let h = params.vol_dims.z;
    if (x >= w || y >= d || z >= h) {
        return;
    }

    let coord = vec3<i32>(i32(x), i32(y), i32(z));
    let dm = textureLoad(density_in, coord, 0);
    let at = textureLoad(albedo_in, coord, 0);

    // Fast path for completely empty zeroed air voxels
    if (dm.r + dm.g + dm.b + at.r + at.g + at.b + at.a < 0.001) {
        textureStore(styled_albedo_out, coord, vec4<f32>(0.0));
        return;
    }

    let y_norm = f32(y) / max(f32(d) - 1.0, 1.0);
    let x0 = at;
    let x1 = vec4<f32>(dm.r, dm.g, dm.b, y_norm);

    // Layer 1: 8 -> 24 (6 vec4 registers, zero memory spills)
    let h1_0 = l1_block(0u, x0, x1);
    let h1_1 = l1_block(1u, x0, x1);
    let h1_2 = l1_block(2u, x0, x1);
    let h1_3 = l1_block(3u, x0, x1);
    let h1_4 = l1_block(4u, x0, x1);
    let h1_5 = l1_block(5u, x0, x1);

    // Layer 2: 24 -> 24 semantic conjunctions (6 vec4 registers)
    let h2_0 = l2_block(0u, h1_0, h1_1, h1_2, h1_3, h1_4, h1_5);
    let h2_1 = l2_block(1u, h1_0, h1_1, h1_2, h1_3, h1_4, h1_5);
    let h2_2 = l2_block(2u, h1_0, h1_1, h1_2, h1_3, h1_4, h1_5);
    let h2_3 = l2_block(3u, h1_0, h1_1, h1_2, h1_3, h1_4, h1_5);
    let h2_4 = l2_block(4u, h1_0, h1_1, h1_2, h1_3, h1_4, h1_5);
    let h2_5 = l2_block(5u, h1_0, h1_1, h1_2, h1_3, h1_4, h1_5);

    // Affine style skip base + nonlinear residual MLP head
    let base_rgba = w_sh[OFF4_SKIP + 4u]
                  + w_sh[OFF4_SKIP + 0u] * x0.r
                  + w_sh[OFF4_SKIP + 1u] * x0.g
                  + w_sh[OFF4_SKIP + 2u] * x0.b
                  + w_sh[OFF4_SKIP + 3u] * x0.a;
    let delta = l3_head(h2_0, h2_1, h2_2, h2_3, h2_4, h2_5);
    let styled = clamp(base_rgba + delta, vec4<f32>(0.0), vec4<f32>(1.0));

    textureStore(styled_albedo_out, coord, styled);
}
