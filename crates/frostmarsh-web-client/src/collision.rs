//! Continuous `∇solid_density` 3D Capsule Physics & Collision Solver for interactive walk mode.

use crate::volume::WorldVolume3D;
use glam::Vec3;

pub const DEFAULT_CAPSULE_RADIUS: f32 = 0.55;
pub const DEFAULT_CAPSULE_HEIGHT: f32 = 1.85;
pub const CAPSULE_EYE_HEIGHT: f32 = 1.58;
pub const GRAVITY: f32 = 18.0;

#[derive(Clone, Copy, Debug)]
pub struct CapsuleStepResult {
    pub pos: Vec3,
    pub vel: Vec3,
    pub grounded: bool,
    pub max_penetration: f32,
}

fn capsule_shell_samples(pos: Vec3, radius: f32, height: f32) -> [Vec3; 26] {
    let r = radius;
    let y_bot = r;
    let y_top = (height - r).max(r + 0.05);
    let y_mid = 0.5 * (y_bot + y_top);
    let d = r * std::f32::consts::FRAC_1_SQRT_2;
    let cap_r = r * 0.7071;
    let cap_dy = r * 0.7071;

    [
        pos + Vec3::new(0.0, 0.0, 0.0),
        pos + Vec3::new(cap_r, y_bot - cap_dy, 0.0),
        pos + Vec3::new(-cap_r, y_bot - cap_dy, 0.0),
        pos + Vec3::new(0.0, y_bot - cap_dy, cap_r),
        pos + Vec3::new(0.0, y_bot - cap_dy, -cap_r),
        pos + Vec3::new(r, y_bot, 0.0),
        pos + Vec3::new(-r, y_bot, 0.0),
        pos + Vec3::new(0.0, y_bot, r),
        pos + Vec3::new(0.0, y_bot, -r),
        pos + Vec3::new(d, y_bot, d),
        pos + Vec3::new(-d, y_bot, d),
        pos + Vec3::new(d, y_bot, -d),
        pos + Vec3::new(-d, y_bot, -d),
        pos + Vec3::new(r, y_mid, 0.0),
        pos + Vec3::new(-r, y_mid, 0.0),
        pos + Vec3::new(0.0, y_mid, r),
        pos + Vec3::new(0.0, y_mid, -r),
        pos + Vec3::new(r, y_top, 0.0),
        pos + Vec3::new(-r, y_top, 0.0),
        pos + Vec3::new(0.0, y_top, r),
        pos + Vec3::new(0.0, y_top, -r),
        pos + Vec3::new(cap_r, y_top + cap_dy, 0.0),
        pos + Vec3::new(-cap_r, y_top + cap_dy, 0.0),
        pos + Vec3::new(0.0, y_top + cap_dy, cap_r),
        pos + Vec3::new(0.0, y_top + cap_dy, -cap_r),
        pos + Vec3::new(0.0, height, 0.0),
    ]
}

pub fn measure_capsule_penetration(
    volume: &WorldVolume3D,
    pos: Vec3,
    radius: f32,
    height: f32,
) -> f32 {
    let mut max_pen = 0.0f32;
    for q in capsule_shell_samples(pos, radius, height) {
        let d = volume.sample_solid(q);
        if d > 0.5 {
            let g_len = volume.sample_solid_grad(q).length().max(0.55);
            let pen = (d - 0.5) / g_len;
            max_pen = max_pen.max(pen);
        }
    }
    max_pen
}

pub fn step_capsule_3d(
    volume: &WorldVolume3D,
    mut pos: Vec3,
    mut vel: Vec3,
    radius: f32,
    height: f32,
    dt: f32,
) -> CapsuleStepResult {
    let substeps = ((dt / 0.008).ceil() as usize).clamp(1, 8);
    let sub_dt = dt / substeps as f32;
    let mut grounded = false;

    let min_x = radius + 1.0;
    let max_x = volume.width as f32 - radius - 1.0;
    let min_z = radius + 1.0;
    let max_z = volume.height as f32 - radius - 1.0;

    for _ in 0..substeps {
        vel.y -= GRAVITY * sub_dt;
        pos += vel * sub_dt;

        pos.x = pos.x.clamp(min_x, max_x);
        pos.z = pos.z.clamp(min_z, max_z);
        pos.y = pos.y.clamp(0.5, volume.depth as f32 - height - 0.1);

        for _iter in 0..10 {
            let mut worst_pen = 0.0f32;
            let mut best_n = Vec3::Y;

            let samples = capsule_shell_samples(pos, radius, height);
            for (idx, &q) in samples.iter().enumerate() {
                let d = volume.sample_solid(q);
                if d > 0.5 {
                    let grad = volume.sample_solid_grad(q);
                    let g_len = grad.length();
                    let fallback = if idx < 5 {
                        Vec3::Y
                    } else if idx >= 21 {
                        -Vec3::Y
                    } else {
                        (Vec3::new(pos.x, q.y, pos.z) - q).normalize_or(Vec3::Y)
                    };
                    let outward_n = if g_len > 1e-3 {
                        -grad / g_len
                    } else {
                        fallback
                    };
                    let pen = (d - 0.5) / g_len.max(0.55);
                    if pen > worst_pen {
                        worst_pen = pen;
                        best_n = outward_n;
                    }
                }
            }

            if worst_pen <= 1e-4 {
                break;
            }

            let push = worst_pen.min(0.35);
            pos += best_n * push;

            if best_n.y >= 0.55 {
                grounded = true;
                if vel.y < 0.0 {
                    vel.y = 0.0;
                }
            }

            let vn = vel.dot(best_n);
            if vn < 0.0 {
                vel -= best_n * vn;
            }
        }
    }

    let max_penetration = measure_capsule_penetration(volume, pos, radius, height);
    CapsuleStepResult {
        pos,
        vel,
        grounded,
        max_penetration,
    }
}
