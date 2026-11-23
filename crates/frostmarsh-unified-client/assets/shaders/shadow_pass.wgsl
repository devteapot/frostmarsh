// 2048x2048 Directional Shadow Map Pass (camera-relative orthographic frustum)

struct SceneUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    vp: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    cam_pos: vec4<f32>,
    cam_fwd: vec4<f32>,
    origin_time: vec4<f32>,
    viewport: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_col: vec4<f32>,
    sky_col: vec4<f32>,
    bounce_col: vec4<f32>,
    fog_col: vec4<f32>,
    point_pos: array<vec4<f32>, 8>,
    point_col: array<vec4<f32>, 8>,
};

struct MeshInstance {
    pos_yaw: vec4<f32>,
    scale_pitch: vec4<f32>,
    tint_glow: vec4<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var<storage, read> instances: array<MeshInstance>;

@vertex
fn vs_mesh_shadow(
    @location(0) pos: vec3<f32>,
    @location(1) _normal: vec3<f32>,
    @location(2) _color_mat: vec4<f32>,
    @location(3) _extra: vec4<f32>,
    @builtin(instance_index) iid: u32,
) -> @builtin(position) vec4<f32> {
    let inst = instances[iid];
    let local_p = pos * inst.scale_pitch.xyz;

    let cr = cos(inst.flags.z);
    let sr = sin(inst.flags.z);
    let p_rolled = vec3<f32>(local_p.x * cr - local_p.y * sr, local_p.x * sr + local_p.y * cr, local_p.z);

    let cp = cos(inst.scale_pitch.w);
    let sp = sin(inst.scale_pitch.w);
    let p_pitched = vec3<f32>(p_rolled.x, p_rolled.y * cp - p_rolled.z * sp, p_rolled.y * sp + p_rolled.z * cp);

    let cy = cos(inst.pos_yaw.w);
    let sy = sin(inst.pos_yaw.w);
    let world_p = inst.pos_yaw.xyz + vec3<f32>(
        p_pitched.x * cy - p_pitched.z * sy,
        p_pitched.y,
        p_pitched.x * sy + p_pitched.z * cy,
    );
    return u.light_vp * vec4<f32>(world_p, 1.0);
}
