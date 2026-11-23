//! Free-Fly / Orbit Perspective Camera for Interactive & Headless Showcase Views.

use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Camera3D {
    pub pos: Vec3,
    pub yaw: f32,   // radians, 0 = +X, PI/2 = +Z
    pub pitch: f32, // radians, clamped [-1.45, 1.45]
    pub fov_y: f32, // radians
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraMatrices {
    pub view: Mat4,
    pub proj: Mat4,
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

impl Camera3D {
    pub fn look_at(pos: Vec3, target: Vec3, fov_deg: f32) -> Self {
        let dir = (target - pos).normalize_or(Vec3::NEG_Z);
        let pitch = dir.y.clamp(-0.99, 0.99).asin();
        let yaw = dir.z.atan2(dir.x);
        Self {
            pos,
            yaw,
            pitch,
            fov_y: fov_deg.to_radians(),
            near: 0.25,
            far: 420.0,
        }
    }

    /// View 1: Wide Arched Stone Bridge + Alpine Mountain Waterfall + Hero on & under bridge + Wildfire
    pub fn preset_bridge_waterfall() -> Self {
        Self::look_at(
            Vec3::new(107.5, 12.3, 153.5),
            Vec3::new(133.5, 7.5, 113.5),
            50.0,
        )
    }

    /// View 2: Dramatic Vaulted Basalt Cave Mouth + Glowing Magma/Crystal Interior + Hero at entrance
    pub fn preset_cave_mouth() -> Self {
        Self::look_at(
            Vec3::new(111.5, 6.8, 147.2),
            Vec3::new(86.5, 6.2, 156.0),
            50.0,
        )
    }

    pub fn forward(&self) -> Vec3 {
        let cp = self.pitch.cos();
        Vec3::new(self.yaw.cos() * cp, self.pitch.sin(), self.yaw.sin() * cp).normalize()
    }

    pub fn matrices(&self, aspect: f32) -> CameraMatrices {
        let forward = self.forward();
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward).normalize_or(Vec3::Y);
        let view = glam::camera::rh::view::look_to_mat4(self.pos, forward, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(
            self.fov_y,
            aspect.max(0.1),
            self.near,
            self.far,
        );
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();
        CameraMatrices {
            view,
            proj,
            view_proj,
            inv_view_proj,
            forward,
            right,
            up,
        }
    }
}
