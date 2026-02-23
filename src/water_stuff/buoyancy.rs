// // src/water/buoyancy.rs
// //
// // CPU-side buoyancy system.
// //
// // Each physics tick, this system:
// //   1. Samples the height field at each voxel position that intersects the
// //      waterline (or below it).
// //   2. Accumulates upward force proportional to submerged volume.
// //   3. Accumulates torque from the asymmetric buoyancy distribution.
// //   4. Applies those forces to the ship's RigidBody (adapt to your physics
// //      backend — examples for both Avian and Rapier are provided).
// //
// // NOTE: This runs on CPU from a readback buffer. For a wallpaper app the
// // latency of one frame is usually fine. For tighter coupling, inject forces
// // into the compute pipeline instead (see §5.3 of the paper).
//
// use bevy::prelude::*;
// use super::{DOMAIN_SIZE, GRID_SIZE, resources::ObstacleMask};
//
// // ─── Resources ────────────────────────────────────────────────────────────────
//
// /// Holds a CPU-readable copy of the height field, updated via async readback
// /// or synchronous copy (depending on your Bevy/wgpu setup).
// ///
// /// For a wallpaper app where exact timing isn't critical, you can simply
// /// update this from a render-world → main-world channel with 1-frame latency.
// #[derive(Resource, Default)]
// pub struct HeightFieldReadback {
//     /// Row-major f32 array, GRID_SIZE × GRID_SIZE.
//     pub data: Vec<f32>,
// }
//
// impl HeightFieldReadback {
//     pub fn new() -> Self {
//         Self {
//             data: vec![0.0f32; (GRID_SIZE * GRID_SIZE) as usize],
//         }
//     }
//
//     /// Sample height (in metres) at a world-space (x, z) position.
//     /// Returns 0 if out of domain.
//     pub fn sample(&self, world_x: f32, world_z: f32) -> f32 {
//         let half = DOMAIN_SIZE * 0.5;
//         let u    = (world_x + half) / DOMAIN_SIZE;
//         let v    = (world_z + half) / DOMAIN_SIZE;
//         if u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0 { return 0.0; }
//         let gx  = (u * GRID_SIZE as f32) as usize;
//         let gy  = (v * GRID_SIZE as f32) as usize;
//         self.data[gy * GRID_SIZE as usize + gx]
//     }
//
//     /// Bilinear sample for smoother buoyancy.
//     pub fn sample_bilinear(&self, world_x: f32, world_z: f32) -> f32 {
//         let half = DOMAIN_SIZE * 0.5;
//         let u    = (world_x + half) / DOMAIN_SIZE * GRID_SIZE as f32;
//         let v    = (world_z + half) / DOMAIN_SIZE * GRID_SIZE as f32;
//
//         let x0 = (u.floor() as usize).min(GRID_SIZE as usize - 1);
//         let y0 = (v.floor() as usize).min(GRID_SIZE as usize - 1);
//         let x1 = (x0 + 1).min(GRID_SIZE as usize - 1);
//         let y1 = (y0 + 1).min(GRID_SIZE as usize - 1);
//         let fx = u.fract();
//         let fy = v.fract();
//
//         let gs = GRID_SIZE as usize;
//         let h00 = self.data[y0 * gs + x0];
//         let h10 = self.data[y0 * gs + x1];
//         let h01 = self.data[y1 * gs + x0];
//         let h11 = self.data[y1 * gs + x1];
//
//         let top = h00 * (1.0 - fx) + h10 * fx;
//         let bot = h01 * (1.0 - fx) + h11 * fx;
//         top * (1.0 - fy) + bot * fy
//     }
// }
//
// // ─── Voxel Buoyancy Data ──────────────────────────────────────────────────────
//
// /// Attach to your ship entity.  Populated by your voxel system.
// ///
// /// Each entry is (world_x, world_y_bottom, world_z, voxel_volume_m3).
// /// world_y_bottom is the *bottom* face of the voxel in world space.
// #[derive(Component)]
// pub struct BuoyancyVoxels {
//     /// (world_x, world_y_bottom, world_z, volume_m3)
//     pub voxels: Vec<(f32, f32, f32, f32)>,
//     /// Water density (kg/m³). Salt water ≈ 1025.
//     pub water_density: f32,
// }
//
// impl Default for BuoyancyVoxels {
//     fn default() -> Self {
//         Self {
//             voxels:        Vec::new(),
//             water_density: 1025.0,
//         }
//     }
// }
//
// // ─── Buoyancy System ─────────────────────────────────────────────────────────
//
// /// Output of the buoyancy calculation, applied to physics each tick.
// #[derive(Component, Default)]
// pub struct BuoyancyOutput {
//     /// Net upward force in Newtons.
//     pub force_n:  Vec3,
//     /// Net torque in N·m around the ship's centre of mass.
//     pub torque_nm: Vec3,
// }
//
// /// Run this in your physics schedule (e.g. `FixedUpdate` or Avian's solver schedule).
// ///
// /// Reads the (possibly 1-frame-old) CPU height field readback and computes
// /// buoyancy forces for each ship voxel.
// pub fn buoyancy_system(
//     height: Res<HeightFieldReadback>,
//     mut query: Query<(&Transform, &BuoyancyVoxels, &mut BuoyancyOutput)>,
// ) {
//     let g = 9.81_f32;
//
//     for (transform, voxels, mut output) in &mut query {
//         let ship_centre = transform.translation;
//         let mut force  = Vec3::ZERO;
//         let mut torque = Vec3::ZERO;
//
//         for &(wx, wy_bot, wz, vol) in &voxels.voxels {
//             // Rotate voxel position from ship-local to world space
//             // (if your voxels are stored in local space).
//             // Assuming world space here for simplicity.
//
//             let water_height = height.sample_bilinear(wx, wz);
//             let voxel_height = 0.5; // approximate voxel half-height — set from your voxel size
//
//             // Depth of submersion = how far below the waterline the voxel top is.
//             let voxel_top    = wy_bot + voxel_height * 2.0;
//             let submerged_fraction = ((water_height - wy_bot) / (voxel_top - wy_bot)).clamp(0.0, 1.0);
//             let submerged_vol = vol * submerged_fraction;
//
//             if submerged_vol <= 0.0 { continue; }
//
//             // Archimedes' principle: F = ρ·g·V_sub  (upward)
//             let f_magnitude = voxels.water_density * g * submerged_vol;
//             let f_voxel     = Vec3::new(0.0, f_magnitude, 0.0);
//
//             // Torque = r × F  where r is voxel position relative to CoM
//             let r = Vec3::new(wx, wy_bot, wz) - ship_centre;
//             torque += r.cross(f_voxel);
//             force  += f_voxel;
//         }
//
//         output.force_n   = force;
//         output.torque_nm = torque;
//     }
// }
//
// /// Update the obstacle mask from the ship's submerged voxels each frame.
// /// Call this after your ship physics update so the mask reflects the new pose.
// pub fn update_obstacle_mask(
//     mut mask: ResMut<ObstacleMask>,
//     query: Query<(&Transform, &BuoyancyVoxels)>,
// ) {
//     mask.clear();
//     for (transform, voxels) in &query {
//         let voxel_positions: Vec<(f32, f32)> = voxels.voxels
//             .iter()
//             // Only include voxels near the waterline for the 2-D obstacle mask.
//             .filter(|&&(_, wy, _, _)| wy.abs() < 2.0)
//             .map(|&(wx, _, wz, _)| (wx, wz))
//             .collect();
//         mask.rasterize_voxels(&voxel_positions);
//     }
// }
