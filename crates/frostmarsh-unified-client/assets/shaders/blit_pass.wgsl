// Stage B: 1-Tap Crisp Nearest-Neighbor Blit (art_res -> 1080p / 4K output texture)

struct BlitUniforms {
    art_res: vec2<f32>,
    out_res: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: BlitUniforms;
@group(0) @binding(1) var art_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pts[vid], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / u.out_res;
    let tc = clamp(vec2<i32>(floor(uv * u.art_res)), vec2<i32>(0, 0), vec2<i32>(u.art_res) - vec2<i32>(1, 1));
    var col = textureLoad(art_tex, tc, 0).rgb;

    // Subtle cinematic vignette
    let dv = uv - vec2<f32>(0.5, 0.5);
    col *= 1.0 - 0.14 * dot(dv, dv);

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
