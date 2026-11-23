// Crisp Native-Resolution GPU 2D Action-MMO HUD & Live Tactical Minimap Pass
// Draws colored UI panels, HP/Mana/Stamina/XP bars, bitmap font glyphs, overhead bars,
// floating combat popups, and the live 256×256 A+ Tactical Minimap texture in one pass!

struct HudUniforms {
    screen_size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> hud: HudUniforms;
@group(0) @binding(1) var minimap_tex: texture_2d<f32>;
@group(0) @binding(2) var minimap_samp: sampler;

struct HudVertIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    // mode: 0.0 = solid colored UI quad/glyph, 1.0 = sample minimap texture * color
    @location(3) mode: f32,
};

struct HudVertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) mode: f32,
};

@vertex
fn vs_main(v: HudVertIn) -> HudVertOut {
    var out: HudVertOut;
    let ndc_x = (v.pos.x / max(hud.screen_size.x, 1.0)) * 2.0 - 1.0;
    let ndc_y = 1.0 - (v.pos.y / max(hud.screen_size.y, 1.0)) * 2.0;
    out.clip_pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    out.mode = v.mode;
    return out;
}

@fragment
fn fs_main(in: HudVertOut) -> @location(0) vec4<f32> {
    if (in.mode > 0.5) {
        let tex_c = textureSample(minimap_tex, minimap_samp, in.uv);
        return vec4<f32>(tex_c.rgb * in.color.rgb, tex_c.a * in.color.a);
    }
    return in.color;
}
