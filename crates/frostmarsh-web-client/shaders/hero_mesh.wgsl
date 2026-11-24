// Wave 18A: Rim-Lit 5-Archetype 3D Parametric GPU Hero & Live Projectile Orb Shader (WebGPU Linear HDR)
// Features:
//   - View-dependent team-tinted Fresnel rim lighting (frost-cyan for Team 0, ember-gold for Team 1)
//   - Crisp emissive core boosts on weapons, Skywing pinions, Tidekin tridents & spinning Burrower drills
//   - Fast 6-tap 3D volumetric sun shadows, 3D cave AO, and style-aware local emissive bounce.

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,         // xyz = cam_pos, w = sim_time
    sun_dir: vec4<f32>,         // xyz = sun_dir, w = exposure
    sun_col: vec4<f32>,         // rgb = sun_col, w = style_mode (0..3)
    sky_top: vec4<f32>,         // rgb = sky_top, w = stylized
    sky_horizon: vec4<f32>,     // rgb = sky_horizon, w = neural_style_strength
    vol_dims: vec4<f32>,        // xyz = (W, D, H), w = 1.0 / D
    brick_dims: vec4<f32>,      // xyz = (bx, by, bz), w = brick_size
    viewport: vec4<f32>,        // xy = (w, h), zw = (1/w, 1/h)
    player_cutaway: vec4<f32>,  // xyz = capsule_pos, w = cutaway_enabled
    style_weights_a: vec4<f32>,
    style_weights_b: vec4<f32>,
    style_weights_c: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var density_tex: texture_3d<f32>;
@group(0) @binding(2) var albedo_tex: texture_3d<f32>;
@group(0) @binding(3) var brick_tex: texture_3d<f32>;
@group(0) @binding(4) var vol_samp: sampler;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color_rough: vec4<f32>,
    @location(3) flags: vec4<f32>,
    @location(4) inst_pos_yaw: vec4<f32>,
    @location(5) inst_faction_scale: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) albedo_rough: vec4<f32>,
    @location(3) emissive: f32,
    @location(4) @interpolate(flat) rim_color: vec3<f32>,
    @location(5) anim_flags: vec2<f32>, // x = wing_flap_weight, y = secondary_anim (1=hover, 2=drill_spin)
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let yaw = in.inst_pos_yaw.w;
    let c = cos(yaw);
    let s = sin(yaw);
    let sc = in.inst_faction_scale.w;
    let t = u.cam_pos.w;

    var anim_p = in.pos;
    var anim_n = in.normal;
    let phase = t * 7.2 + in.inst_pos_yaw.x * 0.31 + in.inst_pos_yaw.z * 0.27;

    // Wing-flap articulation (flags.z = flap weight for flying Skywings)
    if (abs(in.flags.z) > 0.01) {
        let flap = sin(phase) * 0.44 * in.flags.z;
        let side = select(-1.0, 1.0, anim_p.x >= 0.0);
        let pivot_x = side * 0.22;
        let pivot_y = 1.22;
        let rx = anim_p.x - pivot_x;
        let ry = anim_p.y - pivot_y;
        let fc = cos(flap * side);
        let fs = sin(flap * side);
        anim_p.x = pivot_x + rx * fc - ry * fs;
        anim_p.y = pivot_y + rx * fs + ry * fc;
        anim_p.z -= abs(anim_p.x) * 0.07 * cos(phase) * in.flags.z;
        let nrx = anim_n.x * fc - anim_n.y * fs;
        let nry = anim_n.x * fs + anim_n.y * fc;
        anim_n = normalize(vec3<f32>(nrx, nry, anim_n.z));
    }

    // Secondary archetype animation (flags.w: 1.0 = Skywing hover bob, 2.0 = Burrower bore-drill spin)
    if (in.flags.w > 1.5) {
        let spin = t * 11.0 + in.inst_pos_yaw.x * 0.5;
        let sc_s = cos(spin);
        let ss_s = sin(spin);
        let dx = anim_p.x - 0.38;
        let dy = anim_p.y - 0.88;
        anim_p.x = 0.38 + dx * sc_s - dy * ss_s;
        anim_p.y = 0.88 + dx * ss_s + dy * sc_s;
        anim_n = normalize(vec3<f32>(
            anim_n.x * sc_s - anim_n.y * ss_s,
            anim_n.x * ss_s + anim_n.y * sc_s,
            anim_n.z
        ));
    } else if (in.flags.w > 0.5) {
        anim_p.y += -sin(phase) * 0.08;
    }

    let lp = anim_p * sc;
    let wp = vec3<f32>(
        lp.x * c + lp.z * s,
        lp.y,
        -lp.x * s + lp.z * c
    ) + in.inst_pos_yaw.xyz;

    let wn = normalize(vec3<f32>(
        anim_n.x * c + anim_n.z * s,
        anim_n.y,
        -anim_n.x * s + anim_n.z * c
    ));

    let base_col = mix(in.color_rough.rgb, in.color_rough.rgb * in.inst_faction_scale.rgb, in.flags.x);

    // Team-tinted Fresnel rim color (frost-cyan for Team 0 blue/cyan, ember-gold for Team 1 red/amber)
    let is_frost_team = in.inst_faction_scale.b > in.inst_faction_scale.r;
    let rim_col = select(vec3<f32>(1.18, 0.52, 0.08), vec3<f32>(0.16, 0.90, 1.24), is_frost_team);

    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    out.world_normal = wn;
    out.albedo_rough = vec4<f32>(base_col, in.color_rough.a);
    out.emissive = in.flags.y;
    out.rim_color = rim_col;
    out.anim_flags = in.flags.zw;
    return out;
}

fn sample_density(p: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(density_tex, vol_samp, p / u.vol_dims.xyz, 0.0);
}

fn sample_albedo(p: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(albedo_tex, vol_samp, p / u.vol_dims.xyz, 0.0);
}

fn blackbody_ramp(temp: f32, tint: vec3<f32>, style_id: i32) -> vec3<f32> {
    let t = clamp(temp, 0.0, 1.0);
    if (style_id == 2) {
        let low = vec3<f32>(0.14, 0.78, 1.00);
        let mid = vec3<f32>(1.00, 0.18, 0.74);
        let hi  = vec3<f32>(1.00, 0.96, 0.80);
        let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
        return mix(bb, tint * 1.35, 0.35) * (t * t * 3.8);
    }
    if (style_id == 3) {
        let low = vec3<f32>(0.04, 0.68, 0.36);
        let mid = vec3<f32>(0.22, 1.00, 0.60);
        let hi  = vec3<f32>(0.86, 1.00, 0.88);
        let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
        return mix(bb, tint * 1.35, 0.32) * (t * t * 3.6);
    }
    let low = vec3<f32>(0.95, 0.14, 0.02);
    let mid = vec3<f32>(1.00, 0.52, 0.08);
    let hi  = vec3<f32>(1.00, 0.94, 0.72);
    let bb = select(mix(mid, hi, (t - 0.5) * 2.0), mix(low, mid, t * 2.0), t < 0.5);
    return mix(bb, tint * 1.35, 0.45) * (t * t * 3.8);
}

fn compute_sun_shadow_hero(p: vec3<f32>) -> f32 {
    let l = u.sun_dir.xyz;
    let dims = u.vol_dims.xyz;
    let taps = array<f32, 6>(0.80, 1.95, 3.8, 6.6, 10.8, 16.0);
    var vis = 1.0;
    for (var i = 0; i < 6; i = i + 1) {
        let sp = p + l * taps[i];
        if (sp.y >= dims.y || sp.x <= 0.0 || sp.x >= dims.x || sp.z <= 0.0 || sp.z >= dims.z) {
            break;
        }
        let d = sample_density(sp).r;
        let pen = smoothstep(0.54, 0.22, d);
        vis = min(vis, pen);
        if (vis < 0.04) {
            return 0.0;
        }
    }
    return vis;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let n = normalize(in.world_normal);
    let v = normalize(u.cam_pos.xyz - p);
    let l = u.sun_dir.xyz;
    let h = normalize(l + v);
    let style_id = i32(round(u.sun_col.w));
    let sim_t = u.cam_pos.w;

    let lin_albedo = pow(clamp(in.albedo_rough.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
    if (in.emissive > 0.90) {
        let ndv_orb = max(dot(n, v), 0.0);
        let core = mix(vec3<f32>(1.0, 0.98, 0.92), lin_albedo, 0.55);
        let orb_hdr = core * (2.6 + 2.2 * pow(ndv_orb, 1.6)) + lin_albedo * 1.4;
        return vec4<f32>(orb_hdr, 1.0);
    }

    let ndl = max(dot(n, l), 0.0);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 0.0);
    let rough = in.albedo_rough.a;

    let shadow = compute_sun_shadow_hero(p + n * 0.2);
    let roof_occ = sample_density(p + vec3<f32>(0.0, 3.2, 0.0)).r * 0.45
        + sample_density(p + vec3<f32>(0.0, 5.2, 0.0)).r * 0.30;
    let ao = clamp(1.0 - roof_occ, 0.32, 1.0);

    var bounce = vec3<f32>(0.0);
    let offsets = array<vec3<f32>, 4>(
        vec3<f32>(-3.5, 0.2, 1.5),
        vec3<f32>(3.0, 0.2, 0.0),
        vec3<f32>(0.0, 0.2, 3.5),
        vec3<f32>(0.0, 0.2, -3.5)
    );
    for (var i = 0; i < 4; i = i + 1) {
        let s = sample_albedo(p + offsets[i]);
        if (s.a > 0.08) {
            bounce += blackbody_ramp(s.a, s.rgb, style_id) * 0.24;
        }
    }

    let sky_w = n.y * 0.5 + 0.5;
    // Guarantee crisp hero silhouette readability inside shadows & vaulted caverns
    let ambient = max(
        mix(vec3<f32>(0.08, 0.11, 0.18), u.sky_top.rgb * 0.88 + u.sky_horizon.rgb * 0.34, sky_w) * ao,
        vec3<f32>(0.13, 0.16, 0.24),
    );
    let cam_fill = 0.16 * ndv;
    let shininess = mix(115.0, 12.0, rough);
    let spec_w = mix(1.05, 0.06, rough);
    let blade_sheen = smoothstep(0.18, 0.10, rough) * 0.45;
    let spec = pow(ndh, shininess) * (spec_w + blade_sheen) * (0.28 + 0.72 * ndl * (0.30 + 0.70 * shadow));

    // View-dependent team-tinted Fresnel rim lighting (frost-cyan Team 0, ember-gold Team 1)
    let hero_fres = pow(1.0 - ndv, 2.15);
    let top_accent = 0.55 + 0.45 * max(n.y, 0.0);
    let team_rim = in.rim_color * hero_fres * top_accent * 1.12;
    let env_rim = pow(1.0 - ndv, 3.0) * (u.sky_horizon.rgb * 0.26 * ao + bounce * 0.48);

    // Crisp emissive core boosts on weapons, Skywing pinions, Tidekin tridents & spinning Burrower drills
    var accent_glow = vec3<f32>(0.0);
    if (in.emissive > 0.04) {
        let is_drill = step(1.5, in.anim_flags.y);
        let is_pinion = smoothstep(0.55, 0.95, in.anim_flags.x);
        let is_weapon_core = smoothstep(0.52, 0.82, in.emissive);
        let pulse_freq = select(4.8, 12.0, is_drill > 0.5);
        let pulse = 0.88 + 0.16 * sin(sim_t * pulse_freq + p.x * 0.45 + p.z * 0.45);
        let core_hot = mix(lin_albedo, vec3<f32>(1.0, 0.97, 0.90), 0.24 + 0.24 * pow(ndv, 1.5) * is_weapon_core);
        let boost = 3.8 + is_pinion * 1.35 + is_drill * 1.65 + is_weapon_core * 0.95;
        accent_glow = core_hot * (in.emissive * boost * pulse);
    }

    let hdr = lin_albedo * (u.sun_col.rgb * (ndl * shadow + cam_fill * 0.35) + ambient + bounce)
        + u.sun_col.rgb * spec + team_rim + env_rim + accent_glow;

    return vec4<f32>(hdr, 1.0);
}
