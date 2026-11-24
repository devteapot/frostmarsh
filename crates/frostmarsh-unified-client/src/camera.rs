//! Rotatable MOBA Camera with Dynamic F5 Presets, Floating-Origin GPU Math & 2048px Texel-Snapped Shadow Frustum.

use glam::{DVec3, Mat4, Vec3};

// =============================================================================
// Wave 34B: Playable Locally-Flat Macro Planet Camera (R = 1,304 m, f64 Geodesic)
//           4-Stage F5 Cycle: Ground3rdPerson -> SkywingFlight -> Stratosphere -> OrbitalGlobe
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MacroPlanetCamMode {
    /// Classic Frostmarsh MOBA angled-overhead tactical camera (~48 deg looking down at your hero!).
    TacticalOverhead,
    Ground3rdPerson,
    SkywingFlight,
    Stratosphere,
    OrbitalGlobe,
}

impl MacroPlanetCamMode {
    pub fn next(self) -> Self {
        match self {
            Self::TacticalOverhead => Self::Ground3rdPerson,
            Self::Ground3rdPerson => Self::SkywingFlight,
            Self::SkywingFlight => Self::Stratosphere,
            Self::Stratosphere => Self::OrbitalGlobe,
            Self::OrbitalGlobe => Self::TacticalOverhead,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::TacticalOverhead => "OVERHEAD",
            Self::Ground3rdPerson => "GROUND",
            Self::SkywingFlight => "FLIGHT",
            Self::Stratosphere => "STRATO",
            Self::OrbitalGlobe => "ORBIT",
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct MacroPlanetCameraMatrices {
    /// Exact `f64` planet-centered eye position (meters).
    pub eye_f64: [f64; 3],
    /// Exact `f64` planet-centered look target (meters).
    pub target_f64: [f64; 3],
    /// Exact `f64` radial/camera up unit vector.
    pub up_f64: [f64; 3],
    /// Camera-relative view matrix (camera sits strictly at `Vec3::ZERO`!).
    pub view: Mat4,
    pub proj: Mat4,
    /// Camera-relative view-projection matrix (`proj * view`, camera at `Vec3::ZERO`).
    pub vp: Mat4,
    pub inv_vp: Mat4,
    pub focal_px: f32,
}

impl MacroPlanetCameraMatrices {
    /// Convert to `CameraMatrices` (origin = `Vec3::ZERO`, `vp` is camera-relative) so `hud.rs`
    /// can project `world_f64_to_camera_rel_f32(world_pos_f64, eye_f64)` with zero `f32` jitter!
    pub fn to_hud_camera_matrices(&self) -> CameraMatrices {
        let fwd_d = (DVec3::from_array(self.target_f64) - DVec3::from_array(self.eye_f64))
            .normalize_or(DVec3::NEG_Z);
        let fwd = Vec3::new(fwd_d.x as f32, fwd_d.y as f32, fwd_d.z as f32);
        CameraMatrices {
            origin: Vec3::ZERO,
            rel_eye: Vec3::ZERO,
            fwd,
            view: self.view,
            proj: self.proj,
            vp: self.vp,
            light_vp: Mat4::IDENTITY,
            focal_px: self.focal_px,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MacroPlanetCamera {
    pub mode: MacroPlanetCamMode,
    pub orbit_yaw: f64,
    pub orbit_pitch: f64,
    /// Distance from hero (Ground/Flight/Strato) or from planet center (OrbitalGlobe, >= 1450m).
    pub orbit_dist: f64,
    pub smoothed_up: DVec3,
    pub smoothed_fwd: DVec3,
    pub eye_f64: DVec3,
    pub target_f64: DVec3,
    pub up_f64: DVec3,
    pub initialized: bool,
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
    pub exposure: f32,
}

impl MacroPlanetCamera {
    pub fn new(r_surf: f64) -> Self {
        let r = r_surf.max(100.0);
        let eye = DVec3::new(0.0, 0.0, r + 28.0);
        Self {
            mode: MacroPlanetCamMode::TacticalOverhead,
            orbit_yaw: 0.42,
            orbit_pitch: 0.78,
            orbit_dist: 38.0,
            smoothed_up: DVec3::Z,
            smoothed_fwd: DVec3::X,
            eye_f64: eye,
            target_f64: DVec3::new(0.0, 0.0, r + 1.5),
            up_f64: DVec3::Z,
            initialized: false,
            fov_y_deg: 44.0,
            near: 0.4,
            far: 3200.0,
            exposure: 1.06,
        }
    }

    /// Raycast screen pixel `(sx, sy)` onto the hero's local horizontal ground tangent plane
    /// `(p - hero_rel) · hero_up = 0`, returning the 3D tangent aim vector from hero to cursor!
    #[allow(clippy::too_many_arguments)]
    pub fn screen_to_tangent_aim_f64(
        &self,
        mats: &MacroPlanetCameraMatrices,
        hero_pos: DVec3,
        hero_up: DVec3,
        sx: f32,
        sy: f32,
        sw: f32,
        sh: f32,
    ) -> Option<DVec3> {
        let ndc_x = (sx / sw.max(1.0)) * 2.0 - 1.0;
        let ndc_y = 1.0 - (sy / sh.max(1.0)) * 2.0;
        let p_near = mats.inv_vp * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let p_far = mats.inv_vp * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        if p_near.w.abs() < 1e-5 || p_far.w.abs() < 1e-5 {
            return None;
        }
        let ro = p_near.truncate() / p_near.w;
        let rf = p_far.truncate() / p_far.w;
        let rd_f = (rf - ro).normalize_or_zero();
        if rd_f.length_squared() < 0.25 {
            return None;
        }
        let ro_d = DVec3::new(ro.x as f64, ro.y as f64, ro.z as f64);
        let rd_d = DVec3::new(rd_f.x as f64, rd_f.y as f64, rd_f.z as f64);
        let hero_rel = hero_pos - DVec3::from_array(mats.eye_f64);
        let up = hero_up.normalize_or(DVec3::Y);
        let denom = rd_d.dot(up);
        if denom.abs() < 1e-4 {
            return None;
        }
        let t = (hero_rel - ro_d).dot(up) / denom;
        if t <= 0.1 || t > 600.0 {
            return None;
        }
        let hit_rel = ro_d + rd_d * t;
        let delta = hit_rel - hero_rel;
        let tang = delta - up * delta.dot(up);
        if tang.length_squared() < 0.04 {
            None
        } else {
            Some(tang)
        }
    }

    pub fn tangent_move_basis_f64(&self, hero_up: DVec3) -> (DVec3, DVec3) {
        let up = hero_up.normalize_or(DVec3::Y);
        let raw_look = (self.target_f64 - self.eye_f64).normalize_or(self.smoothed_fwd);
        let mut fwd = (raw_look - up * raw_look.dot(up)).normalize_or_zero();
        if fwd.length_squared() < 0.15 {
            fwd = (self.smoothed_fwd - up * self.smoothed_fwd.dot(up)).normalize_or_zero();
        }
        if fwd.length_squared() < 0.15 {
            fwd = if up.y.abs() < 0.9 {
                DVec3::Y.cross(up).normalize()
            } else {
                DVec3::X.cross(up).normalize()
            };
        }
        let right = fwd.cross(up).normalize_or(DVec3::X);
        (fwd, right)
    }

    pub fn drag_orbit(&mut self, dx: f32, dy: f32) {
        self.orbit_yaw -= dx as f64 * 0.006;
        self.orbit_pitch = (self.orbit_pitch + dy as f64 * 0.005).clamp(-1.32, 1.32);
    }

    /// Smooth continuous scroll-wheel zoom all the way from Ground (`6 m`) -> Overhead (`38 m`) -> Orbit (`4,600 m`)!
    pub fn zoom(&mut self, delta: f32) {
        let factor = (1.0 - (delta as f64) * 0.045).clamp(0.70, 1.45);
        self.orbit_dist = (self.orbit_dist * factor).clamp(6.0, 4600.0);
        self.mode = if self.orbit_dist < 18.0 {
            MacroPlanetCamMode::Ground3rdPerson
        } else if self.orbit_dist < 62.0 {
            MacroPlanetCamMode::TacticalOverhead
        } else if self.orbit_dist < 260.0 {
            MacroPlanetCamMode::SkywingFlight
        } else if self.orbit_dist < 1350.0 {
            MacroPlanetCamMode::Stratosphere
        } else {
            MacroPlanetCamMode::OrbitalGlobe
        };
    }

    /// Cycle through the 5 signature views via `F5`: OVERHEAD -> GROUND -> FLIGHT -> STRATO -> ORBIT.
    pub fn cycle_mode(&mut self) {
        self.mode = self.mode.next();
        match self.mode {
            MacroPlanetCamMode::TacticalOverhead => {
                self.orbit_dist = 38.0;
                self.orbit_pitch = 0.78;
                self.fov_y_deg = 44.0;
                self.exposure = 1.06;
            }
            MacroPlanetCamMode::Ground3rdPerson => {
                self.orbit_dist = 12.0;
                self.orbit_pitch = 0.22;
                self.fov_y_deg = 56.0;
                self.exposure = 1.06;
            }
            MacroPlanetCamMode::SkywingFlight => {
                self.orbit_dist = 85.0;
                self.orbit_pitch = 0.36;
                self.fov_y_deg = 58.0;
                self.exposure = 1.08;
            }
            MacroPlanetCamMode::Stratosphere => {
                self.orbit_dist = 550.0;
                self.orbit_pitch = 0.55;
                self.fov_y_deg = 60.0;
                self.exposure = 1.14;
            }
            MacroPlanetCamMode::OrbitalGlobe => {
                self.orbit_dist = 3250.0;
                self.orbit_pitch = 0.24;
                self.fov_y_deg = 52.0;
                self.exposure = 1.22;
            }
        }
    }

    /// Smooth `f64` geodesic follow & seamless Ground -> Flight -> Stratosphere -> Orbit transition.
    pub fn update_follow(
        &mut self,
        hero_world_pos: DVec3,
        hero_tangent_fwd: DVec3,
        r_surf: f64,
        dt: f32,
    ) {
        let hero_up = hero_world_pos.normalize_or(DVec3::Z);
        let dt_c = (dt as f64).clamp(0.001, 0.1);
        let alpha_up = if !self.initialized { 1.0 } else { 1.0 - (-8.0 * dt_c).exp() };
        let alpha_pos = if !self.initialized { 1.0 } else { 1.0 - (-10.0 * dt_c).exp() };

        // Geodesic slerp of radial UP across seams & poles
        let dot_u = self.smoothed_up.dot(hero_up).clamp(-1.0, 1.0);
        if !self.initialized {
            self.smoothed_up = hero_up;
        } else if dot_u < 0.99999 {
            let omega = dot_u.acos();
            let sin_o = omega.sin();
            if sin_o > 1e-5 {
                let w0 = ((1.0 - alpha_up) * omega).sin() / sin_o;
                let w1 = (alpha_up * omega).sin() / sin_o;
                self.smoothed_up = (self.smoothed_up * w0 + hero_up * w1).normalize_or(hero_up);
            } else {
                self.smoothed_up = self.smoothed_up.lerp(hero_up, alpha_up).normalize_or(hero_up);
            }
        } else {
            self.smoothed_up = hero_up;
        }

        // Parallel-transport tangent forward perpendicular to `smoothed_up`
        let proj_cur = (self.smoothed_fwd - self.smoothed_up * self.smoothed_fwd.dot(self.smoothed_up))
            .normalize_or_zero();
        let proj_tgt = (hero_tangent_fwd - self.smoothed_up * hero_tangent_fwd.dot(self.smoothed_up))
            .normalize_or_zero();
        let fallback_fwd = if self.smoothed_up.y.abs() < 0.9 {
            DVec3::Y.cross(self.smoothed_up).normalize()
        } else {
            DVec3::X.cross(self.smoothed_up).normalize()
        };
        if !self.initialized {
            self.smoothed_fwd = if proj_tgt.length_squared() > 0.25 { proj_tgt } else { fallback_fwd };
        } else if proj_cur.length_squared() > 0.25 {
            self.smoothed_fwd = proj_cur;
        } else if proj_tgt.length_squared() > 0.25 {
            self.smoothed_fwd = proj_tgt;
        } else {
            self.smoothed_fwd = fallback_fwd;
        }

        let tang_r = self.smoothed_fwd.cross(self.smoothed_up).normalize_or(DVec3::X);
        let look_dir = (self.smoothed_fwd * self.orbit_yaw.cos()
            + tang_r * self.orbit_yaw.sin())
        .normalize_or(self.smoothed_fwd);

        let (des_eye, des_tgt, des_up, fov, near, far, exp) = match self.mode {
            MacroPlanetCamMode::TacticalOverhead => {
                // Classic Frostmarsh angled-from-above MOBA camera (~48 deg pitch looking right down at your hero!)
                let dist = self.orbit_dist.clamp(18.0, 62.0);
                let up_h = dist * 0.78;
                let back_d = dist * 0.64;
                let eye = hero_world_pos + self.smoothed_up * up_h - look_dir * back_d;
                let tgt = hero_world_pos + self.smoothed_up * 1.4;
                (eye, tgt, self.smoothed_up, 44.0, 0.4, 3400.0, 1.06)
            }
            MacroPlanetCamMode::Ground3rdPerson => {
                // Classic over-the-shoulder 3rd-person (~12m behind, ~4.5m above hero) on 1m flat ground!
                let back_d = self.orbit_dist.clamp(6.0, 22.0);
                let up_h = (back_d * 0.36).clamp(3.0, 10.0);
                let eye = hero_world_pos + self.smoothed_up * up_h - look_dir * back_d;
                let tgt = hero_world_pos + look_dir * 32.0 + self.smoothed_up * 2.2;
                (eye, tgt, self.smoothed_up, 56.0, 0.25, 2900.0, 1.06)
            }
            MacroPlanetCamMode::SkywingFlight => {
                // Dramatic Skywing flight (~85m above hero) framing coastline, Sky-Island & curving horizon!
                let alt = self.orbit_dist.clamp(36.0, 260.0);
                let back_d = alt * 0.82;
                let eye = hero_world_pos + self.smoothed_up * alt - look_dir * back_d;
                let tgt = hero_world_pos + look_dir * (alt * 1.45) + self.smoothed_up * 8.0;
                (eye, tgt, self.smoothed_up, 58.0, 0.8, 4600.0, 1.08)
            }
            MacroPlanetCamMode::Stratosphere => {
                // High stratosphere (~550m above hero) where sky fades to starry space & limb bends vividly!
                let alt = self.orbit_dist.clamp(260.0, 1349.0);
                let eye_dir = (hero_up - look_dir * (alt * 0.42 / r_surf.max(100.0))).normalize();
                let eye = eye_dir * (r_surf + alt);
                let tgt_dir = (hero_up + look_dir * (alt * 0.65 / r_surf.max(100.0))).normalize();
                let tgt = tgt_dir * r_surf;
                (eye, tgt, eye_dir, 60.0, 4.0, 7200.0, 1.14)
            }
            MacroPlanetCamMode::OrbitalGlobe => {
                // Full orbital space view (~3,250m from center) showing the entire R=1,304m planet & Coriolis storms!
                let cp = self.orbit_pitch.cos();
                let sp = self.orbit_pitch.sin();
                let dir = DVec3::new(self.orbit_yaw.sin() * cp, sp, self.orbit_yaw.cos() * cp).normalize();
                let dist = self.orbit_dist.clamp(2100.0, 4600.0);
                let eye = dir * dist;
                let up_ref = if dir.y.abs() > 0.96 { DVec3::NEG_Z } else { DVec3::Y };
                (eye, DVec3::ZERO, up_ref, 52.0, 25.0, 11000.0, 1.22)
            }
        };

        self.eye_f64 = self.eye_f64.lerp(des_eye, alpha_pos);
        self.target_f64 = self.target_f64.lerp(des_tgt, alpha_pos);
        self.up_f64 = self.up_f64.lerp(des_up, alpha_pos).normalize_or(des_up);
        self.fov_y_deg = fov;
        self.near = near;
        self.far = far;
        self.exposure = exp;
        self.initialized = true;
    }

    pub fn compute_matrices(&self, rt_w: f32, rt_h: f32) -> MacroPlanetCameraMatrices {
        let aspect = (rt_w / rt_h.max(1.0)).max(0.1);
        let fwd_d = (self.target_f64 - self.eye_f64).normalize_or(DVec3::NEG_Z);
        let up_d = self.up_f64.normalize_or(DVec3::Y);
        let safe_up_d = if fwd_d.dot(up_d).abs() > 0.995 {
            if up_d.y.abs() < 0.9 { DVec3::Y } else { DVec3::X }
        } else {
            up_d
        };
        let fwd = Vec3::new(fwd_d.x as f32, fwd_d.y as f32, fwd_d.z as f32).normalize_or(Vec3::NEG_Z);
        let up = Vec3::new(safe_up_d.x as f32, safe_up_d.y as f32, safe_up_d.z as f32).normalize_or(Vec3::Y);

        // Floating-origin camera-relative matrix: eye sits strictly at Vec3::ZERO!
        let view = glam::camera::rh::view::look_at_mat4(Vec3::ZERO, fwd, up);
        let fov_rad = self.fov_y_deg.to_radians();
        let proj = glam::camera::rh::proj::directx::perspective(
            fov_rad,
            aspect,
            self.near.max(0.1),
            self.far.max(100.0),
        );
        let vp = proj * view;
        let inv_vp = vp.inverse();
        let focal_px = (rt_h * 0.5) / (fov_rad * 0.5).tan();
        MacroPlanetCameraMatrices {
            eye_f64: self.eye_f64.to_array(),
            target_f64: self.target_f64.to_array(),
            up_f64: safe_up_d.to_array(),
            view,
            proj,
            vp,
            inv_vp,
            focal_px,
        }
    }

    /// Headless Shot 1: Ground 3rd-Person Hero Shot on locally-flat 1m terrain framing hero, bots & Sky-Island!
    pub fn ground_3rd_person_shot(hero_pos: DVec3, hero_fwd: DVec3, sky_island_dir: DVec3) -> Self {
        let up = hero_pos.normalize_or(DVec3::Z);
        let to_isl = (sky_island_dir - up * sky_island_dir.dot(up)).normalize_or_zero();
        let mut fwd = if to_isl.length_squared() > 0.1 {
            to_isl
        } else {
            (hero_fwd - up * hero_fwd.dot(up)).normalize_or_zero()
        };
        if fwd.length_squared() < 0.15 {
            fwd = if up.y.abs() < 0.9 { DVec3::Y.cross(up).normalize() } else { DVec3::X.cross(up).normalize() };
        }
        let right = fwd.cross(up).normalize_or(DVec3::X);
        let eye = hero_pos + up * 4.8 - fwd * 12.5 + right * 2.2;
        let tgt = hero_pos + fwd * 42.0 + up * 4.5;
        Self {
            mode: MacroPlanetCamMode::Ground3rdPerson,
            orbit_yaw: 0.0,
            orbit_pitch: 0.22,
            orbit_dist: 12.5,
            smoothed_up: up,
            smoothed_fwd: fwd,
            eye_f64: eye,
            target_f64: tgt,
            up_f64: up,
            initialized: true,
            fov_y_deg: 56.0,
            near: 0.25,
            far: 2900.0,
            exposure: 1.06,
        }
    }

    /// Headless Shot 2: Skywing Flight Action Shot (~95m altitude) framing Sky-Island, coastline & curving horizon!
    pub fn skywing_flight_shot(hero_pos: DVec3, hero_fwd: DVec3, sky_island_dir: DVec3, r_surf: f64) -> Self {
        let up = hero_pos.normalize_or(DVec3::Z);
        let to_isl = (sky_island_dir - up * sky_island_dir.dot(up)).normalize_or_zero();
        let mut fwd = if to_isl.length_squared() > 0.1 {
            to_isl
        } else {
            (hero_fwd - up * hero_fwd.dot(up)).normalize_or_zero()
        };
        if fwd.length_squared() < 0.15 {
            fwd = if up.y.abs() < 0.9 { DVec3::Y.cross(up).normalize() } else { DVec3::X.cross(up).normalize() };
        }
        let right = fwd.cross(up).normalize_or(DVec3::X);
        let flight_dir = (up - fwd * (44.0 / r_surf.max(100.0)) - right * (58.0 / r_surf.max(100.0))).normalize();
        let eye = flight_dir * (r_surf + 95.0);
        let tgt = hero_pos + fwd * 68.0 + up * 22.0;
        Self {
            mode: MacroPlanetCamMode::SkywingFlight,
            orbit_yaw: 0.0,
            orbit_pitch: 0.36,
            orbit_dist: 95.0,
            smoothed_up: flight_dir,
            smoothed_fwd: fwd,
            eye_f64: eye,
            target_f64: tgt,
            up_f64: flight_dir,
            initialized: true,
            fov_y_deg: 58.0,
            near: 0.8,
            far: 4600.0,
            exposure: 1.08,
        }
    }

    /// Headless Shot 3: Stratosphere (~550m altitude) — curving planetary limb fading into cosmic starfield!
    pub fn stratosphere_shot(hero_pos: DVec3, hero_fwd: DVec3, r_surf: f64) -> Self {
        let up = hero_pos.normalize_or(DVec3::Z);
        let mut fwd = (hero_fwd - up * hero_fwd.dot(up)).normalize_or_zero();
        if fwd.length_squared() < 0.15 {
            fwd = if up.y.abs() < 0.9 { DVec3::Y.cross(up).normalize() } else { DVec3::X.cross(up).normalize() };
        }
        let right = fwd.cross(up).normalize_or(DVec3::X);
        let strat_dir = (up - fwd * (240.0 / r_surf.max(100.0)) - right * (120.0 / r_surf.max(100.0))).normalize();
        let eye = strat_dir * (r_surf + 550.0);
        let tgt = (up + fwd * (390.0 / r_surf.max(100.0))).normalize() * r_surf;
        Self {
            mode: MacroPlanetCamMode::Stratosphere,
            orbit_yaw: 0.0,
            orbit_pitch: 0.55,
            orbit_dist: 550.0,
            smoothed_up: strat_dir,
            smoothed_fwd: fwd,
            eye_f64: eye,
            target_f64: tgt,
            up_f64: strat_dir,
            initialized: true,
            fov_y_deg: 60.0,
            near: 4.0,
            far: 7200.0,
            exposure: 1.14,
        }
    }

    /// Instant 1-key landmark inspection jump (`F7` / `Alt+1..6`): snaps camera smoothly onto any
    /// planetary landmark without diving through the sphere core!
    pub fn focus_landmark(&mut self, target_pos: DVec3, look_fwd: DVec3, dist: f64, pitch: f64) {
        let up = target_pos.normalize_or(DVec3::Z);
        let mut fwd = (look_fwd - up * look_fwd.dot(up)).normalize_or_zero();
        if fwd.length_squared() < 0.15 {
            fwd = if up.y.abs() < 0.9 {
                DVec3::Y.cross(up).normalize()
            } else {
                DVec3::X.cross(up).normalize()
            };
        }
        let d = dist.clamp(12.0, 240.0);
        self.mode = if d < 18.0 {
            MacroPlanetCamMode::Ground3rdPerson
        } else if d < 62.0 {
            MacroPlanetCamMode::TacticalOverhead
        } else {
            MacroPlanetCamMode::SkywingFlight
        };
        self.orbit_dist = d;
        self.orbit_pitch = pitch.clamp(0.16, 1.15);
        self.orbit_yaw = 0.0;
        self.smoothed_up = up;
        self.smoothed_fwd = fwd;
        let up_h = d * self.orbit_pitch.sin().clamp(0.32, 0.85);
        let back_d = d * self.orbit_pitch.cos().clamp(0.45, 0.92);
        self.eye_f64 = target_pos + up * up_h - fwd * back_d;
        self.target_f64 = target_pos + up * 1.8;
        self.up_f64 = up;
        self.fov_y_deg = if d < 62.0 { 48.0 } else { 56.0 };
        self.near = if d < 62.0 { 0.35 } else { 0.8 };
        self.far = 4200.0;
        self.exposure = 1.07;
        self.initialized = true;
    }

    /// Headless Bonus Shot 5: Angled close-up overlooking Nordic Coastal Village, pier, longboats & bay!
    pub fn village_harbor_shot(village_pos: DVec3, look_dir: DVec3) -> Self {
        let up = village_pos.normalize_or(DVec3::Z);
        let mut fwd = (look_dir - up * look_dir.dot(up)).normalize_or_zero();
        if fwd.length_squared() < 0.15 {
            fwd = if up.y.abs() < 0.9 {
                DVec3::Y.cross(up).normalize()
            } else {
                DVec3::X.cross(up).normalize()
            };
        }
        let right = fwd.cross(up).normalize_or(DVec3::X);
        let eye = village_pos + up * 15.5 - fwd * 26.0 + right * 9.5;
        let tgt = village_pos + fwd * 8.0 + up * 2.8;
        Self {
            mode: MacroPlanetCamMode::TacticalOverhead,
            orbit_yaw: 0.0,
            orbit_pitch: 0.52,
            orbit_dist: 30.0,
            smoothed_up: up,
            smoothed_fwd: fwd,
            eye_f64: eye,
            target_f64: tgt,
            up_f64: up,
            initialized: true,
            fov_y_deg: 50.0,
            near: 0.30,
            far: 3400.0,
            exposure: 1.06,
        }
    }

    /// Headless Bonus Shot 6: Angled close-up overlooking glowing Cavern Grotto or Carved River Valley!
    pub fn cavern_river_shot(site_pos: DVec3, look_dir: DVec3) -> Self {
        let up = site_pos.normalize_or(DVec3::Z);
        let mut fwd = (look_dir - up * look_dir.dot(up)).normalize_or_zero();
        if fwd.length_squared() < 0.15 {
            fwd = if up.y.abs() < 0.9 {
                DVec3::Y.cross(up).normalize()
            } else {
                DVec3::X.cross(up).normalize()
            };
        }
        let right = fwd.cross(up).normalize_or(DVec3::X);
        let eye = site_pos + up * 13.5 - fwd * 20.5 + right * 6.0;
        let tgt = site_pos + fwd * 3.0 + up * 1.8;
        Self {
            mode: MacroPlanetCamMode::TacticalOverhead,
            orbit_yaw: 0.0,
            orbit_pitch: 0.56,
            orbit_dist: 24.0,
            smoothed_up: up,
            smoothed_fwd: fwd,
            eye_f64: eye,
            target_f64: tgt,
            up_f64: up,
            initialized: true,
            fov_y_deg: 52.0,
            near: 0.25,
            far: 3200.0,
            exposure: 1.08,
        }
    }

    /// Headless Shot 4: Full Orbital Globe (~3,250m) framing spinning R=1,304m planet, Coriolis storms & Terminator!
    pub fn orbital_globe_shot(sun_dir: DVec3, _r_surf: f64) -> Self {
        let sun = sun_dir.normalize_or(DVec3::new(-0.72, 0.36, 0.59).normalize());
        let perp = DVec3::Y.cross(sun).normalize_or(DVec3::Z);
        let cam_dir = (sun * 0.24 + perp * 0.91 + DVec3::Y * 0.26).normalize();
        let dist = 3250.0;
        Self {
            mode: MacroPlanetCamMode::OrbitalGlobe,
            orbit_yaw: cam_dir.x.atan2(cam_dir.z),
            orbit_pitch: cam_dir.y.asin(),
            orbit_dist: dist,
            smoothed_up: DVec3::Y,
            smoothed_fwd: -cam_dir,
            eye_f64: cam_dir * dist,
            target_f64: DVec3::ZERO,
            up_f64: DVec3::Y,
            initialized: true,
            fov_y_deg: 52.0,
            near: 25.0,
            far: 11000.0,
            exposure: 1.24,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlanetCamMode {
    OrbitalGlobe,
    LowOrbitTactical,
    SurfaceHorizon,
}

impl PlanetCamMode {
    pub fn next(self) -> Self {
        match self {
            Self::OrbitalGlobe => Self::LowOrbitTactical,
            Self::LowOrbitTactical => Self::SurfaceHorizon,
            Self::SurfaceHorizon => Self::OrbitalGlobe,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OrbitalGlobe => "ORBITAL",
            Self::LowOrbitTactical => "TACTICAL",
            Self::SurfaceHorizon => "SURFACE",
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct PlanetCameraMatrices {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub view: Mat4,
    pub proj: Mat4,
    pub vp: Mat4,
    pub inv_vp: Mat4,
    pub focal_px: f32,
}

impl PlanetCameraMatrices {
    /// Convert to `CameraMatrices` (floating origin = 0) so 2D HUD world-space projection works identically!
    pub fn to_camera_matrices(&self) -> CameraMatrices {
        let fwd = (self.target - self.eye).normalize_or(Vec3::NEG_Z);
        CameraMatrices {
            origin: Vec3::ZERO,
            rel_eye: self.eye,
            fwd,
            view: self.view,
            proj: self.proj,
            vp: self.vp,
            light_vp: Mat4::IDENTITY,
            focal_px: self.focal_px,
        }
    }
}

/// 6-DOF Geodesic Spherical Planetary Camera (`OrbitalGlobe`, `LowOrbitTactical`, `SurfaceHorizon`).
/// Smoothly slerps `UP = normalize(hero_pos)` across all 12 cube seams & both poles with zero gimbal lock!
#[derive(Clone, Debug)]
pub struct PlanetCamera {
    pub mode: PlanetCamMode,
    pub orbit_yaw: f32,
    pub orbit_pitch: f32,
    pub orbit_dist: f32,
    pub smoothed_up: Vec3,
    pub smoothed_fwd: Vec3,
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
    pub exposure: f32,
}

impl PlanetCamera {
    pub fn new(r_core: f32, depth: f32) -> Self {
        let r_surf = r_core + depth * 0.55;
        let eye = Vec3::new(0.0, 6.0, r_surf + 14.0);
        Self {
            mode: PlanetCamMode::SurfaceHorizon,
            orbit_yaw: 0.0,
            orbit_pitch: 0.24,
            orbit_dist: 14.0,
            smoothed_up: Vec3::Z,
            smoothed_fwd: Vec3::X,
            eye,
            target: Vec3::new(0.0, 0.0, r_surf),
            up: Vec3::Z,
            fov_y_deg: 56.0,
            near: 0.35,
            far: 950.0,
            exposure: 1.12,
        }
    }

    /// Return the camera's `(forward, right)` unit vectors projected onto the tangent plane perpendicular to `hero_up`.
    pub fn tangent_move_basis(&self, hero_up: Vec3) -> (Vec3, Vec3) {
        let up = hero_up.normalize_or(Vec3::Y);
        let raw_look = (self.target - self.eye).normalize_or(self.smoothed_fwd);
        let mut fwd = (raw_look - up * raw_look.dot(up)).normalize_or_zero();
        if fwd.length_squared() < 0.15 {
            fwd = (self.smoothed_fwd - up * self.smoothed_fwd.dot(up)).normalize_or_zero();
        }
        if fwd.length_squared() < 0.15 {
            fwd = if up.y.abs() < 0.9 {
                Vec3::Y.cross(up).normalize()
            } else {
                Vec3::X.cross(up).normalize()
            };
        }
        let right = fwd.cross(up).normalize_or(Vec3::X);
        (fwd, right)
    }

    pub fn drag_orbit(&mut self, dx: f32, dy: f32) {
        self.orbit_yaw -= dx * 0.006;
        self.orbit_pitch = (self.orbit_pitch + dy * 0.005).clamp(-1.35, 1.35);
    }

    pub fn zoom(&mut self, delta: f32, r_core: f32, depth: f32) {
        let r_outer = r_core + depth;
        let (min_d, max_d) = match self.mode {
            PlanetCamMode::OrbitalGlobe => (r_outer * 1.45, r_outer * 4.2),
            PlanetCamMode::LowOrbitTactical => (18.0, r_outer * 1.8),
            PlanetCamMode::SurfaceHorizon => (6.0, 42.0),
        };
        self.orbit_dist = (self.orbit_dist - delta * 2.2).clamp(min_d, max_d);
    }

    pub fn cycle_mode(&mut self, r_core: f32, depth: f32) {
        self.mode = self.mode.next();
        let r_outer = r_core + depth;
        match self.mode {
            PlanetCamMode::OrbitalGlobe => {
                self.orbit_dist = r_outer * 2.45;
                self.fov_y_deg = 44.0;
                self.exposure = 1.22;
            }
            PlanetCamMode::LowOrbitTactical => {
                self.orbit_dist = 46.0;
                self.fov_y_deg = 48.0;
                self.exposure = 1.15;
            }
            PlanetCamMode::SurfaceHorizon => {
                self.orbit_dist = 14.0;
                self.fov_y_deg = 56.0;
                self.exposure = 1.10;
            }
        }
    }

    /// Update 6-DOF geodesic spherical follow & orbit camera with smooth slerp of `UP = normalize(hero_pos)`.
    pub fn update_follow(
        &mut self,
        hero_world_pos: Vec3,
        hero_tangent_fwd: Vec3,
        _sun_dir: Vec3,
        r_core: f32,
        depth: f32,
        dt: f32,
    ) {
        let hero_up = hero_world_pos.normalize_or(Vec3::Z);
        let alpha_up = 1.0 - (-7.5 * dt.clamp(0.001, 0.1)).exp();

        // Geodesic slerp of radial UP vector across cube seams & poles
        let dot_u = self.smoothed_up.dot(hero_up).clamp(-1.0, 1.0);
        if dot_u < 0.9999 {
            let omega = dot_u.acos();
            let sin_o = omega.sin();
            if sin_o > 1e-4 {
                let w0 = ((1.0 - alpha_up) * omega).sin() / sin_o;
                let w1 = (alpha_up * omega).sin() / sin_o;
                self.smoothed_up = (self.smoothed_up * w0 + hero_up * w1).normalize_or(hero_up);
            } else {
                self.smoothed_up = self.smoothed_up.lerp(hero_up, alpha_up).normalize_or(hero_up);
            }
        } else {
            self.smoothed_up = hero_up;
        }

        // Parallel-transport `smoothed_fwd` smoothly across the curved sphere perpendicular to `smoothed_up`
        // (never snaps when strafing A/D or walking backwards S — camera heading stays rock-steady!).
        let proj_cur_fwd =
            (self.smoothed_fwd - self.smoothed_up * self.smoothed_fwd.dot(self.smoothed_up))
                .normalize_or_zero();
        let proj_target_fwd =
            (hero_tangent_fwd - self.smoothed_up * hero_tangent_fwd.dot(self.smoothed_up))
                .normalize_or_zero();
        let fallback_fwd = if self.smoothed_up.y.abs() < 0.9 {
            Vec3::Y.cross(self.smoothed_up).normalize()
        } else {
            Vec3::X.cross(self.smoothed_up).normalize()
        };
        if proj_cur_fwd.length_squared() > 0.25 {
            self.smoothed_fwd = proj_cur_fwd;
        } else if proj_target_fwd.length_squared() > 0.25 {
            self.smoothed_fwd = proj_target_fwd;
        } else {
            self.smoothed_fwd = fallback_fwd;
        }

        let r_outer = r_core + depth;
        let alpha_pos = 1.0 - (-10.0 * dt.clamp(0.001, 0.1)).exp();

        match self.mode {
            PlanetCamMode::OrbitalGlobe => {
                let cp = self.orbit_pitch.cos();
                let sp = self.orbit_pitch.sin();
                let dir = Vec3::new(self.orbit_yaw.sin() * cp, sp, self.orbit_yaw.cos() * cp).normalize();
                let dist = self.orbit_dist.clamp(r_outer * 1.45, r_outer * 4.2);
                let des_eye = dir * dist;
                self.eye = self.eye.lerp(des_eye, alpha_pos);
                self.target = self.target.lerp(Vec3::ZERO, alpha_pos);
                let up_ref = if dir.y.abs() > 0.96 { Vec3::NEG_Z } else { Vec3::Y };
                self.up = self.up.lerp(up_ref, alpha_pos).normalize_or(Vec3::Y);
                self.fov_y_deg = 44.0;
                self.near = 1.0;
                self.far = 950.0;
                self.exposure = 1.22;
            }
            PlanetCamMode::LowOrbitTactical => {
                let tang_r = self.smoothed_fwd.cross(self.smoothed_up).normalize_or(Vec3::X);
                let yaw_dir = self.smoothed_fwd * self.orbit_yaw.cos() + tang_r * self.orbit_yaw.sin();
                let dist = self.orbit_dist.clamp(22.0, r_outer * 1.65);
                let des_eye = hero_world_pos + self.smoothed_up * (dist * 0.78) - yaw_dir * (dist * 0.62);
                self.eye = self.eye.lerp(des_eye, alpha_pos);
                self.target = self.target.lerp(hero_world_pos + self.smoothed_up * 2.0, alpha_pos);
                self.up = self.smoothed_up;
                self.fov_y_deg = 48.0;
                self.near = 0.5;
                self.far = 700.0;
                self.exposure = 1.15;
            }
            PlanetCamMode::SurfaceHorizon => {
                // Orbitable 3rd-person over-the-shoulder geodesic camera behind your hero!
                let back_d = self.orbit_dist.clamp(8.0, 28.0);
                let tang_r = self.smoothed_fwd.cross(self.smoothed_up).normalize_or(Vec3::X);
                let look_dir = (self.smoothed_fwd * self.orbit_yaw.cos()
                    + tang_r * self.orbit_yaw.sin())
                .normalize_or(self.smoothed_fwd);
                let des_eye = hero_world_pos + self.smoothed_up * (back_d * 0.44).clamp(4.0, 11.0)
                    - look_dir * back_d;
                let des_tgt = hero_world_pos + look_dir * 22.0 + self.smoothed_up * 1.4;
                self.eye = self.eye.lerp(des_eye, alpha_pos);
                self.target = self.target.lerp(des_tgt, alpha_pos);
                self.up = self.smoothed_up;
                self.fov_y_deg = 56.0;
                self.near = 0.35;
                self.far = 650.0;
                self.exposure = 1.12;
            }
        }
    }

    pub fn compute_matrices(&self, rt_w: f32, rt_h: f32) -> PlanetCameraMatrices {
        let aspect = (rt_w / rt_h.max(1.0)).max(0.1);
        let up_n = self.up.normalize_or(Vec3::Y);
        let look = (self.target - self.eye).normalize_or(Vec3::NEG_Z);
        let safe_up = if look.dot(up_n).abs() > 0.995 {
            if up_n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X }
        } else {
            up_n
        };
        let view = glam::camera::rh::view::look_at_mat4(self.eye, self.target, safe_up);
        let fov_rad = self.fov_y_deg.to_radians();
        let proj = glam::camera::rh::proj::directx::perspective(
            fov_rad,
            aspect,
            self.near.max(0.1),
            self.far.max(100.0),
        );
        let vp = proj * view;
        let inv_vp = vp.inverse();
        let focal_px = (rt_h * 0.5) / (fov_rad * 0.5).tan();
        PlanetCameraMatrices {
            eye: self.eye,
            target: self.target,
            up: safe_up,
            view,
            proj,
            vp,
            inv_vp,
            focal_px,
        }
    }

    /// Hero Shot 1: Dramatic 3/4 Full-Globe Orbital Terminator View
    pub fn orbital_terminator_shot(sun_dir: Vec3, r_outer: f32) -> Self {
        let sun = sun_dir.normalize_or(Vec3::new(-0.78, 0.28, 0.56).normalize());
        // Position camera ~78° from sun_dir so ~58% is brilliantly sunlit, ~42% is luminous night side,
        // framing the sweeping Day/Night Terminator right across the globe!
        let perp = Vec3::Y.cross(sun).normalize_or(Vec3::Z);
        let cam_dir = (sun * 0.24 + perp * 0.92 + Vec3::Y * 0.26).normalize();
        let dist = (r_outer * 2.48).max(176.0);
        Self {
            mode: PlanetCamMode::OrbitalGlobe,
            orbit_yaw: cam_dir.x.atan2(cam_dir.z),
            orbit_pitch: cam_dir.y.asin(),
            orbit_dist: dist,
            smoothed_up: Vec3::Y,
            smoothed_fwd: -cam_dir,
            eye: cam_dir * dist,
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_deg: 44.0,
            near: 1.0,
            far: 950.0,
            exposure: 1.24,
        }
    }

    /// Hero Shot 2: Low-Altitude 6-DOF Curved Surface Horizon Action View
    pub fn surface_horizon_battle_shot(focus_pos: Vec3, focus_fwd: Vec3) -> Self {
        let up = focus_pos.normalize_or(Vec3::Z);
        let mut fwd = (focus_fwd - up * focus_fwd.dot(up)).normalize_or_zero();
        if fwd.length_squared() < 0.25 {
            fwd = if up.y.abs() < 0.9 {
                Vec3::Y.cross(up).normalize()
            } else {
                Vec3::X.cross(up).normalize()
            };
        }
        let eye = focus_pos + up * 5.4 - fwd * 14.5;
        let target = focus_pos + fwd * 22.0 + up * 1.0;
        Self {
            mode: PlanetCamMode::SurfaceHorizon,
            orbit_yaw: 0.0,
            orbit_pitch: 0.0,
            orbit_dist: 16.0,
            smoothed_up: up,
            smoothed_fwd: fwd,
            eye,
            target,
            up,
            fov_y_deg: 56.0,
            near: 0.35,
            far: 600.0,
            exposure: 1.12,
        }
    }

    /// Showcase View 3: North Polar Glacier Cap & Orbital Sky-Ring
    pub fn polar_sky_ring_shot(r_outer: f32) -> Self {
        let dir = Vec3::new(-0.34, 0.86, 0.38).normalize();
        let dist = (r_outer * 2.42).max(172.0);
        Self {
            mode: PlanetCamMode::OrbitalGlobe,
            orbit_yaw: -0.7,
            orbit_pitch: 1.0,
            orbit_dist: dist,
            smoothed_up: Vec3::NEG_Z,
            smoothed_fwd: -dir,
            eye: dir * dist,
            target: Vec3::new(0.0, 4.0, 0.0),
            up: Vec3::new(0.0, 0.0, -1.0),
            fov_y_deg: 44.0,
            near: 1.0,
            far: 950.0,
            exposure: 1.18,
        }
    }

    /// Showcase View 4: Night-Side Glowing Volcanic Rift & Live Beams + Sunlit Crescent Limb
    pub fn night_volcanic_rift_shot(sun_dir: Vec3, r_outer: f32) -> Self {
        let sun = sun_dir.normalize_or(Vec3::new(-0.78, 0.28, 0.56).normalize());
        let perp = Vec3::Y.cross(sun).normalize_or(Vec3::Z);
        // Position mostly on the night hemisphere (~132° from sun) with a glowing Rayleigh crescent on the rim!
        let dir = (-sun * 0.68 + perp * 0.64 + Vec3::Y * 0.34).normalize();
        let dist = (r_outer * 2.38).max(170.0);
        Self {
            mode: PlanetCamMode::OrbitalGlobe,
            orbit_yaw: dir.x.atan2(dir.z),
            orbit_pitch: dir.y.asin(),
            orbit_dist: dist,
            smoothed_up: Vec3::Y,
            smoothed_fwd: -dir,
            eye: dir * dist,
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_deg: 44.0,
            near: 1.0,
            far: 950.0,
            exposure: 1.32,
        }
    }
}

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
