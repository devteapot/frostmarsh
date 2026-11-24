//! Rotatable MOBA Camera with Dynamic F5 Presets, Floating-Origin GPU Math & 2048px Texel-Snapped Shadow Frustum.

use glam::{Mat4, Vec3};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CameraPreset {
    Tactical,
    ActionCloseUp,
    Overview,
}

impl CameraPreset {
    pub fn next(self) -> Self {
        match self {
            Self::Tactical => Self::ActionCloseUp,
            Self::ActionCloseUp => Self::Overview,
            Self::Overview => Self::Tactical,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Tactical => "TACTICAL",
            Self::ActionCloseUp => "ACTION",
            Self::Overview => "OVERVIEW",
        }
    }

    /// Returns `(dist, pitch_rad, fov_y_rad)` tuned for true-3D `Z=16..64` or L4 arenas.
    #[allow(dead_code)]
    pub fn params(self, z16_active: bool) -> (f32, f32, f32) {
        self.params_for_depth(z16_active, 16)
    }

    /// Returns `(dist, pitch_rad, fov_y_rad)` scaled smoothly for `Z=16/32` AND tall 3-strata `Z=64` worlds!
    pub fn params_for_depth(self, z16_active: bool, depth: usize) -> (f32, f32, f32) {
        if z16_active && depth >= 48 {
            match self {
                Self::Tactical => (88.0, 0.34, 42.0_f32.to_radians()),
                Self::ActionCloseUp => (28.0, 0.30, 42.0_f32.to_radians()),
                Self::Overview => (138.0, 0.42, 44.0_f32.to_radians()),
            }
        } else if z16_active {
            match self {
                Self::Tactical => (40.0, 0.56, 38.0_f32.to_radians()),
                Self::ActionCloseUp => (24.0, 0.32, 40.0_f32.to_radians()),
                Self::Overview => (82.0, 0.68, 40.0_f32.to_radians()),
            }
        } else {
            match self {
                Self::Tactical => (46.0, 0.64, 38.0_f32.to_radians()),
                Self::ActionCloseUp => (26.0, 0.38, 40.0_f32.to_radians()),
                Self::Overview => (96.0, 0.72, 40.0_f32.to_radians()),
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct MobaCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub fov_y: f32,
    pub ortho: bool,
}

impl Default for MobaCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 0.8, 0.0),
            yaw: 0.42,
            pitch: 0.78, // ~45 deg classic MOBA isometric angle
            dist: 76.0,
            fov_y: 38.0_f32.to_radians(),
            ortho: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CameraMatrices {
    pub origin: Vec3,
    pub rel_eye: Vec3,
    pub fwd: Vec3,
    pub view: Mat4,
    pub proj: Mat4,
    pub vp: Mat4,
    pub light_vp: Mat4,
    pub focal_px: f32,
}

impl MobaCamera {
    pub fn floating_origin(&self) -> Vec3 {
        Vec3::new(
            (self.target.x / 64.0).floor() * 64.0,
            0.0,
            (self.target.z / 64.0).floor() * 64.0,
        )
    }

    #[allow(deprecated)]
    pub fn compute_matrices(&self, rt_w: f32, rt_h: f32, sun_dir: Vec3) -> CameraMatrices {
        let origin = self.floating_origin();
        let rel_target = self.target - origin;
        let aspect = (rt_w / rt_h.max(1.0)).max(0.1);

        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();

        let offset_dir = Vec3::new(sy * cp, sp, cy * cp).normalize();
        let eye_dist = if self.ortho { 160.0 } else { self.dist };
        let rel_eye = rel_target + offset_dir * eye_dist;
        let fwd = (rel_target - rel_eye).normalize();

        let view = Mat4::look_at_rh(rel_eye, rel_target, Vec3::Y);
        let half_h = self.dist * (self.fov_y * 0.5).tan();

        let (proj, focal_px) = if self.ortho {
            let half_w = half_h * aspect;
            let p = Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, 1.0, 420.0);
            (p, rt_h / (2.0 * half_h))
        } else {
            let near = (self.dist * 0.04).max(0.5);
            let far = self.dist * 6.0 + 340.0;
            let p = Mat4::perspective_rh(self.fov_y, aspect, near, far);
            (p, (rt_h * 0.5) / (self.fov_y * 0.5).tan())
        };

        let vp = proj * view;

        // 2048px texel-snapped directional shadow frustum covering the visible Frostmarsh arena (up to Z=64!)
        let sh_extent = (self.dist * 1.15).clamp(36.0, 128.0);
        let sun_n = sun_dir.normalize_or(Vec3::Y);
        let light_up = if sun_n.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
        let raw_light_view = Mat4::look_at_rh(rel_target + sun_n * 140.0, rel_target, light_up);
        let texel_world = (sh_extent * 2.0) / 2048.0;
        let ls_origin = raw_light_view.transform_point3(Vec3::ZERO);
        let snap_x = (ls_origin.x / texel_world).round() * texel_world - ls_origin.x;
        let snap_y = (ls_origin.y / texel_world).round() * texel_world - ls_origin.y;
        let light_proj = Mat4::orthographic_rh(
            -sh_extent + snap_x,
            sh_extent + snap_x,
            -sh_extent + snap_y,
            sh_extent + snap_y,
            5.0,
            380.0,
        );
        let light_vp = light_proj * raw_light_view;

        CameraMatrices {
            origin,
            rel_eye,
            fwd,
            view,
            proj,
            vp,
            light_vp,
            focal_px,
        }
    }

    /// Unproject a screen pixel `(sx, sy)` onto horizontal plane `y = plane_y`, returning world `(wx, wz)`.
    pub fn screen_to_world_xz(
        &self,
        mats: &CameraMatrices,
        sx: f32,
        sy: f32,
        sw: f32,
        sh: f32,
        plane_y: f32,
    ) -> Option<(f32, f32)> {
        let ndc_x = (sx / sw.max(1.0)) * 2.0 - 1.0;
        let ndc_y = 1.0 - (sy / sh.max(1.0)) * 2.0;
        let inv_vp = mats.vp.inverse();
        let p_near = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let p_far = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        if p_near.w.abs() < 1e-5 || p_far.w.abs() < 1e-5 {
            return None;
        }
        let ro = p_near.truncate() / p_near.w;
        let rf = p_far.truncate() / p_far.w;
        let rd = rf - ro;
        if rd.y.abs() < 1e-5 {
            return None;
        }
        let t = (plane_y - mats.origin.y - ro.y) / rd.y;
        if t < 0.0 {
            return None;
        }
        let hit = ro + rd * t + mats.origin;
        Some((hit.x, hit.z))
    }
}
