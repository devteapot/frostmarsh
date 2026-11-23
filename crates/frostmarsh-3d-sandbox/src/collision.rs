//! Continuous `∇solid_density` 3D Capsule Physics & Collision Solver.
//!
//! Operates directly on the continuous trilinear `solid_density` scalar field (`0.5` isosurface)
//! without triangle meshes or heightfields, enabling true 3D multi-level traversal:
//! walking across arched bridges, walking underneath bridge archways, entering vaulted caves,
//! and sliding smoothly tangent along angled rock walls.

use crate::volume::WorldVolume3D;
use glam::Vec3;
use serde::Serialize;

pub const DEFAULT_CAPSULE_RADIUS: f32 = 0.55;
pub const DEFAULT_CAPSULE_HEIGHT: f32 = 1.85;
pub const GRAVITY: f32 = 18.0;

#[derive(Clone, Copy, Debug)]
pub struct CapsuleStepResult {
    pub pos: Vec3,
    pub vel: Vec3,
    pub grounded: bool,
    pub max_penetration: f32,
    pub ceiling_contact: bool,
    pub wall_contact: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CapsuleAuditMetrics {
    pub bridge_walk_ok: bool,
    pub bridge_min_y: f32,
    pub bridge_max_y: f32,
    pub bridge_max_penetration_m: f32,
    pub under_bridge_walk_ok: bool,
    pub under_bridge_min_clearance_m: f32,
    pub under_bridge_max_penetration_m: f32,
    pub cave_traverse_ok: bool,
    pub cave_min_ceiling_clearance_m: f32,
    pub cave_max_penetration_m: f32,
    pub wall_slide_45deg_ok: bool,
    pub wall_slide_max_penetration_m: f32,
    pub wall_slide_min_tangent_speed: f32,
}

/// Evaluate capsule shell sample points (relative to foot base `pos`).
/// Uses bottom cap, lower/mid/upper cylindrical rings, and top dome cap so thin bridges,
/// vaulted cave ceilings, and 45° walls are all sampled cleanly without snagging.
fn capsule_shell_samples(pos: Vec3, radius: f32, height: f32) -> [Vec3; 26] {
    let r = radius;
    let y_bot = r;
    let y_top = (height - r).max(r + 0.05);
    let y_mid = 0.5 * (y_bot + y_top);
    let d = r * std::f32::consts::FRAC_1_SQRT_2;
    let cap_r = r * 0.7071;
    let cap_dy = r * 0.7071;

    [
        // Bottom pole + 4 lower hemisphere points (5)
        pos + Vec3::new(0.0, 0.0, 0.0),
        pos + Vec3::new(cap_r, y_bot - cap_dy, 0.0),
        pos + Vec3::new(-cap_r, y_bot - cap_dy, 0.0),
        pos + Vec3::new(0.0, y_bot - cap_dy, cap_r),
        pos + Vec3::new(0.0, y_bot - cap_dy, -cap_r),
        // Lower cylinder ring (8)
        pos + Vec3::new(r, y_bot, 0.0),
        pos + Vec3::new(-r, y_bot, 0.0),
        pos + Vec3::new(0.0, y_bot, r),
        pos + Vec3::new(0.0, y_bot, -r),
        pos + Vec3::new(d, y_bot, d),
        pos + Vec3::new(-d, y_bot, d),
        pos + Vec3::new(d, y_bot, -d),
        pos + Vec3::new(-d, y_bot, -d),
        // Mid-waist ring (4)
        pos + Vec3::new(r, y_mid, 0.0),
        pos + Vec3::new(-r, y_mid, 0.0),
        pos + Vec3::new(0.0, y_mid, r),
        pos + Vec3::new(0.0, y_mid, -r),
        // Upper cylinder ring (4)
        pos + Vec3::new(r, y_top, 0.0),
        pos + Vec3::new(-r, y_top, 0.0),
        pos + Vec3::new(0.0, y_top, r),
        pos + Vec3::new(0.0, y_top, -r),
        // Top hemisphere + top pole (5)
        pos + Vec3::new(cap_r, y_top + cap_dy, 0.0),
        pos + Vec3::new(-cap_r, y_top + cap_dy, 0.0),
        pos + Vec3::new(0.0, y_top + cap_dy, cap_r),
        pos + Vec3::new(0.0, y_top + cap_dy, -cap_r),
        pos + Vec3::new(0.0, height, 0.0),
    ]
}

/// Measure maximum solid penetration (in meters) across the capsule shell.
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

/// Measure vertical open air clearance (in meters) directly above `pos.y` at `(pos.x, pos.z)`.
pub fn measure_vertical_clearance(volume: &WorldVolume3D, pos: Vec3) -> f32 {
    let step = 0.1;
    let mut y = pos.y + 0.15;
    while y < volume.depth as f32 - 0.5 {
        if volume.sample_solid(Vec3::new(pos.x, y, pos.z)) >= 0.5 {
            return (y - pos.y).max(0.0);
        }
        y += step;
    }
    (volume.depth as f32 - pos.y).max(0.0)
}

/// Step a 3D hero capsule `(pos = foot center, vel)` through the continuous `solid_density` field
/// using trilinear density sampling, analytical `∇solid_density` pushout, tangent velocity
/// projection, and gravity/floor settling.
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
    let mut ceiling_contact = false;
    let mut wall_contact = false;

    let min_x = radius + 1.0;
    let max_x = volume.width as f32 - radius - 1.0;
    let min_z = radius + 1.0;
    let max_z = volume.height as f32 - radius - 1.0;

    for _ in 0..substeps {
        // Apply gravity
        vel.y -= GRAVITY * sub_dt;
        pos += vel * sub_dt;

        // Keep inside horizontal world bounds
        pos.x = pos.x.clamp(min_x, max_x);
        pos.z = pos.z.clamp(min_z, max_z);
        pos.y = pos.y.clamp(0.5, volume.depth as f32 - height - 0.1);

        // Iterative Gauss-Seidel pushout along analytical -∇solid_density
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

            // Push capsule out along outward normal
            let push = worst_pen.min(0.35);
            pos += best_n * push;

            // Classify contact and project velocity onto tangent plane (zero wall-slide deadlocks!)
            if best_n.y >= 0.55 {
                grounded = true;
                if vel.y < 0.0 {
                    vel.y = 0.0;
                }
            } else if best_n.y <= -0.55 {
                ceiling_contact = true;
            } else {
                wall_contact = true;
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
        ceiling_contact,
        wall_contact,
    }
}

/// Run the full 4-scenario continuous 3D capsule physics audit on the showcase world volume.
pub fn run_capsule_audit(volume: &WorldVolume3D) -> CapsuleAuditMetrics {
    let r = DEFAULT_CAPSULE_RADIUS;
    let h = DEFAULT_CAPSULE_HEIGHT;
    let dt = 1.0 / 60.0;

    // 1. Walk across the arched stone bridge from X=102 -> X=156 along Z=128
    let mut pos = Vec3::new(102.0, 8.8, 128.0);
    let mut vel = Vec3::ZERO;
    // Settle onto bridge first
    for _ in 0..30 {
        let res = step_capsule_3d(volume, pos, vel, r, h, dt);
        pos = res.pos;
        vel = res.vel;
    }
    let mut bridge_min_y = f32::MAX;
    let mut bridge_max_y = f32::MIN;
    let mut bridge_max_pen = 0.0f32;
    for _ in 0..680 {
        vel.x = 5.0;
        vel.z = (128.0 - pos.z) * 2.0;
        let res = step_capsule_3d(volume, pos, vel, r, h, dt);
        pos = res.pos;
        vel = res.vel;
        bridge_min_y = bridge_min_y.min(pos.y);
        bridge_max_y = bridge_max_y.max(pos.y);
        bridge_max_pen = bridge_max_pen.max(res.max_penetration);
        if pos.x >= 155.0 {
            break;
        }
    }
    let bridge_walk_ok =
        pos.x >= 154.0 && bridge_min_y >= 8.0 && bridge_max_y <= 9.6 && bridge_max_pen < 0.05;

    // 2. Walk cleanly underneath the West bridge archway along X=113.5 from Z=114 -> Z=142
    let mut pos = Vec3::new(113.5, 4.3, 114.0);
    let mut vel = Vec3::ZERO;
    for _ in 0..25 {
        let res = step_capsule_3d(volume, pos, vel, r, h, dt);
        pos = res.pos;
        vel = res.vel;
    }
    let mut under_min_clear = f32::MAX;
    let mut under_max_pen = 0.0f32;
    for _ in 0..420 {
        vel.z = 4.5;
        vel.x = (113.5 - pos.x) * 2.0;
        let res = step_capsule_3d(volume, pos, vel, r, h, dt);
        pos = res.pos;
        vel = res.vel;
        if (pos.z - 128.0).abs() <= 4.0 {
            under_min_clear = under_min_clear.min(measure_vertical_clearance(volume, pos));
        }
        under_max_pen = under_max_pen.max(res.max_penetration);
        if pos.z >= 141.0 {
            break;
        }
    }
    let under_bridge_walk_ok =
        pos.z >= 140.0 && pos.y < 5.0 && under_min_clear >= 3.0 && under_max_pen < 0.05;

    // 3. Enter and traverse the vaulted basalt cave from entrance (103, 4.5, 150.5) -> inner chamber (81, 4.5, 159.5)
    let waypoints = [
        Vec3::new(97.0, 4.3, 152.5),
        Vec3::new(89.5, 4.3, 155.0),
        Vec3::new(81.5, 4.3, 159.5),
    ];
    let mut pos = Vec3::new(103.0, 4.6, 150.5);
    let mut vel = Vec3::ZERO;
    for _ in 0..25 {
        let res = step_capsule_3d(volume, pos, vel, r, h, dt);
        pos = res.pos;
        vel = res.vel;
    }
    let mut wp_idx = 0usize;
    let mut cave_min_clear = f32::MAX;
    let mut cave_max_pen = 0.0f32;
    let mut cave_ceiling_hit = false;
    for _ in 0..500 {
        let target = waypoints[wp_idx];
        let to_t = Vec3::new(target.x - pos.x, 0.0, target.z - pos.z);
        if to_t.length() < 1.2 {
            if wp_idx + 1 < waypoints.len() {
                wp_idx += 1;
            } else {
                break;
            }
        }
        let dir = to_t.normalize_or(Vec3::NEG_X);
        vel.x = dir.x * 4.2;
        vel.z = dir.z * 4.2;
        let res = step_capsule_3d(volume, pos, vel, r, h, dt);
        pos = res.pos;
        vel = res.vel;
        cave_max_pen = cave_max_pen.max(res.max_penetration);
        cave_ceiling_hit |= res.ceiling_contact;
        if pos.x <= 97.5 {
            cave_min_clear = cave_min_clear.min(measure_vertical_clearance(volume, pos));
        }
    }
    let cave_traverse_ok = wp_idx == waypoints.len() - 1
        && (pos.x - 81.5).hypot(pos.z - 159.5) < 2.2
        && cave_min_clear >= 2.5
        && !cave_ceiling_hit
        && cave_max_pen < 0.05;

    // 4. Slide smoothly tangent along the 45° basalt rock wall near (X=107, Z=105)
    // Wall plane: (x - 107) + (z - 105) = 0, outward normal (+0.7071, 0, +0.7071).
    // Drive capsule straight in -X (45° into wall!) and verify smooth tangent slide (-X, +Z) with pen < 0.05m!
    let mut pos = Vec3::new(110.0, 9.0, 103.5);
    let mut vel = Vec3::ZERO;
    for _ in 0..35 {
        let res = step_capsule_3d(volume, pos, vel, r, h, dt);
        pos = res.pos;
        vel = res.vel;
    }
    let mut slide_max_pen = 0.0f32;
    let mut slide_min_tan_speed = f32::MAX;
    let start_z = pos.z;
    let mut contacted_steps = 0usize;
    for _ in 0..160 {
        // Push directly -X (45° into the angled wall!) while preserving tangent momentum
        vel.x = -4.5;
        vel.z = vel.z * 0.25;
        let res = step_capsule_3d(volume, pos, vel, r, h, dt);
        pos = res.pos;
        vel = res.vel;
        slide_max_pen = slide_max_pen.max(res.max_penetration);
        if res.wall_contact {
            contacted_steps += 1;
            let horiz_speed = vel.x.hypot(vel.z);
            slide_min_tan_speed = slide_min_tan_speed.min(horiz_speed);
        }
    }
    let slid_dist_z = pos.z - start_z;
    if contacted_steps == 0 {
        slide_min_tan_speed = 0.0;
    }
    let wall_slide_45deg_ok = contacted_steps >= 20
        && slid_dist_z >= 1.5
        && slide_min_tan_speed >= 2.2
        && slide_max_pen < 0.05;

    CapsuleAuditMetrics {
        bridge_walk_ok,
        bridge_min_y,
        bridge_max_y,
        bridge_max_penetration_m: bridge_max_pen,
        under_bridge_walk_ok,
        under_bridge_min_clearance_m: under_min_clear,
        under_bridge_max_penetration_m: under_max_pen,
        cave_traverse_ok,
        cave_min_ceiling_clearance_m: cave_min_clear,
        cave_max_penetration_m: cave_max_pen,
        wall_slide_45deg_ok,
        wall_slide_max_penetration_m: slide_max_pen,
        wall_slide_min_tangent_speed: slide_min_tan_speed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::WorldVolume3D;

    #[test]
    fn test_continuous_capsule_3d_all_scenarios() {
        let vol = WorldVolume3D::generate_showcase();
        let audit = run_capsule_audit(&vol);
        assert!(
            audit.bridge_walk_ok,
            "Bridge walk failed: min_y={:.3}, max_y={:.3}, max_pen={:.4}",
            audit.bridge_min_y, audit.bridge_max_y, audit.bridge_max_penetration_m
        );
        assert!(
            audit.under_bridge_walk_ok,
            "Under-bridge walk failed: min_clearance={:.3}, max_pen={:.4}",
            audit.under_bridge_min_clearance_m, audit.under_bridge_max_penetration_m
        );
        assert!(
            audit.cave_traverse_ok,
            "Cave traversal failed: min_clearance={:.3}, max_pen={:.4}",
            audit.cave_min_ceiling_clearance_m, audit.cave_max_penetration_m
        );
        assert!(
            audit.wall_slide_45deg_ok,
            "45-degree wall slide failed: min_tan_speed={:.3}, max_pen={:.4}",
            audit.wall_slide_min_tangent_speed, audit.wall_slide_max_penetration_m
        );
    }
}
