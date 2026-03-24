use bevy::app::{App, Plugin, Startup, Update};
use bevy::light::NotShadowCaster;
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy::mesh::{Indices, PrimitiveTopology};
use num_complex::Complex32;
use phastft::planner::Direction;
use rand::Rng;

const GRID_SIZE: usize = 256;

pub struct BigWatering;
impl Plugin for BigWatering {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_ocean);
        app.add_systems(Update, update_ocean);
    }
}

struct OceanConfig {
    grid_size: usize,
    patch_length: f32,
    wind_speed: f32,
    wind_dir: Vec2,
    amplitude: f32,
    gravity: f32,
    fetch: f32,
    gamma: f32,
}

impl Default for OceanConfig {
    fn default() -> Self {
        Self {
            grid_size: GRID_SIZE,
            patch_length: 256.0,
            wind_speed: 10.0,
            wind_dir: Vec2::new(1.0, 0.0).normalize(),
            amplitude: 0.01,
            gravity: 9.81,
            fetch: 100_000.0,
            gamma: 3.3,
        }
    }
}

/// Mesh vertex spacing relative to height field grid points.
/// 1.0 = same resolution as grid, 0.5 = 2x denser, 2.0 = half resolution.
const MESH_STEP: f32 = 1.0/8.0;

#[derive(Component)]
#[require(Transform)]
struct OceanMesh;

#[derive(Resource)]
struct OceanSimulation {
    config: OceanConfig,
    h0: Vec<Complex32>,
    h0_conj: Vec<Complex32>,
    omega: Vec<f32>,
    spectrum: Vec<Complex32>,
    heights: Vec<f32>,
    normals: Vec<Vec3>,
    row_scratch: Vec<Complex32>,
}

fn jonswap_spectrum(kx: f32, kz: f32, config: &OceanConfig) -> f32 {
    let k_mag = (kx * kx + kz * kz).sqrt();
    if k_mag < 1e-8 {
        return 0.0;
    }

    let g = config.gravity;
    let omega = (g * k_mag).sqrt();

    // Peak frequency (JONSWAP)
    let omega_p = 22.0 * (g * g / (config.wind_speed * config.fetch)).powf(1.0 / 3.0);

    if omega < 1e-8 {
        return 0.0;
    }

    // Pierson-Moskowitz base spectrum
    let alpha = 0.0081;
    let omega_ratio = omega_p / omega;
    let s_pm = (alpha * g * g / omega.powi(5))
        * (-1.25 * omega_ratio.powi(4)).exp();

    // JONSWAP peak enhancement
    let sigma = if omega <= omega_p { 0.07 } else { 0.09 };
    let r = ((omega - omega_p).powi(2)) / (2.0 * sigma * sigma * omega_p * omega_p);
    let gamma_factor = config.gamma.powf((-r).exp());

    let s_jonswap = s_pm * gamma_factor;

    // Convert to k-space: S(k) = S(omega) * d_omega/dk / (2*pi)
    // d_omega/dk = g / (2 * omega) for deep water dispersion
    let d_omega_dk = g / (2.0 * omega);
    let s_k = s_jonswap * d_omega_dk / (2.0 * std::f32::consts::PI);

    // Directional spreading: cos^2(theta) where theta is angle between k and wind
    let k_dir = Vec2::new(kx, kz) / k_mag;
    let cos_theta = k_dir.dot(config.wind_dir);
    let directional = if cos_theta > 0.0 {
        cos_theta * cos_theta
    } else {
        0.0
    };

    (config.amplitude * s_k * directional).max(0.0)
}

fn box_muller(rng: &mut impl Rng) -> Complex32 {
    let u1: f32 = rng.random::<f32>().max(1e-10);
    let u2: f32 = rng.random::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    Complex32::new(r * theta.cos(), r * theta.sin())
}

fn ifft_2d(
    data: &mut [Complex32],
    n: usize,
    scratch: &mut Vec<Complex32>,
) {
    scratch.resize(n, Complex32::new(0.0, 0.0));

    // Row pass
    for z in 0..n {
        let offset = z * n;
        scratch.copy_from_slice(&data[offset..offset + n]);

        phastft::fft_32_interleaved(scratch, Direction::Reverse);

        data[offset..offset + n].copy_from_slice(scratch);
    }

    // Column pass
    for x in 0..n {
        for z in 0..n {
            scratch[z] = data[z * n + x];
        }

        phastft::fft_32_interleaved(scratch, Direction::Reverse);

        for z in 0..n {
            // fftshift sign correction: (-1)^(x+z)
            // Also multiply by N² to undo phastft's 1/N normalization per pass,
            // since Tessendorf expects an unnormalized DFT sum.
            let sign = if (x + z) % 2 == 0 { 1.0 } else { -1.0 };
            data[z * n + x].re = scratch[z].re * sign * (n * n) as f32;
        }
    }
}

fn compute_normals(heights: &[f32], normals: &mut [Vec3], n: usize, patch_length: f32) {
    let dx = patch_length / n as f32;
    for z in 0..n {
        for x in 0..n {
            let left = heights[z * n + (x + n - 1) % n];
            let right = heights[z * n + (x + 1) % n];
            let down = heights[((z + n - 1) % n) * n + x];
            let up = heights[((z + 1) % n) * n + x];

            let dhdx = (right - left) / (2.0 * dx);
            let dhdz = (up - down) / (2.0 * dx);

            normals[z * n + x] = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
        }
    }
}

/// Bilinear interpolation into a flat N×N grid of f32.
fn sample_bilinear(data: &[f32], n: usize, fx: f32, fz: f32) -> f32 {
    let x0 = (fx as usize).min(n - 1);
    let z0 = (fz as usize).min(n - 1);
    let x1 = (x0 + 1).min(n - 1);
    let z1 = (z0 + 1).min(n - 1);
    let tx = fx - x0 as f32;
    let tz = fz - z0 as f32;

    let v00 = data[z0 * n + x0];
    let v10 = data[z0 * n + x1];
    let v01 = data[z1 * n + x0];
    let v11 = data[z1 * n + x1];

    let top = v00 + (v10 - v00) * tx;
    let bot = v01 + (v11 - v01) * tx;
    top + (bot - top) * tz
}

/// Bilinear interpolation into a flat N×N grid of Vec3, returns normalized result.
fn sample_bilinear_normal(data: &[Vec3], n: usize, fx: f32, fz: f32) -> Vec3 {
    let x0 = (fx as usize).min(n - 1);
    let z0 = (fz as usize).min(n - 1);
    let x1 = (x0 + 1).min(n - 1);
    let z1 = (z0 + 1).min(n - 1);
    let tx = fx - x0 as f32;
    let tz = fz - z0 as f32;

    let v00 = data[z0 * n + x0];
    let v10 = data[z0 * n + x1];
    let v01 = data[z1 * n + x0];
    let v11 = data[z1 * n + x1];
    let top = v00 + (v10 - v00) * tx;
    let bot = v01 + (v11 - v01) * tx;
    let result = top + (bot - top) * tz;
    result.normalize_or_zero()
}

fn build_ocean_mesh(n: usize, patch_length: f32, step: f32) -> Mesh {
    let mesh_n = (n as f32 / step) as usize; // vertices per side
    let dx = patch_length / n as f32;
    let offset = -patch_length / 2.0;

    let vertex_count = mesh_n * mesh_n;
    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(vertex_count);

    for z in 0..mesh_n {
        for x in 0..mesh_n {
            let px = offset + x as f32 * step * dx;
            let pz = offset + z as f32 * step * dx;
            positions.push([px, 0.0, pz]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([x as f32 / mesh_n as f32, z as f32 / mesh_n as f32]);
        }
    }

    let mut indices = Vec::with_capacity((mesh_n - 1) * (mesh_n - 1) * 6);
    for z in 0..(mesh_n - 1) {
        for x in 0..(mesh_n - 1) {
            let i = (z * mesh_n + x) as u32;
            let right = i + 1;
            let below = i + mesh_n as u32;
            let below_right = below + 1;
            indices.push(i);
            indices.push(below);
            indices.push(right);
            indices.push(right);
            indices.push(below);
            indices.push(below_right);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn init_ocean(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let config = OceanConfig::default();
    let n = config.grid_size;
    let n2 = n * n;

    let mut h0 = vec![Complex32::new(0.0, 0.0); n2];
    let mut h0_conj = vec![Complex32::new(0.0, 0.0); n2];
    let mut omega = vec![0.0f32; n2];

    let mut rng = rand::rng();
    let l = config.patch_length;
    let g = config.gravity;
    let half_n = (n / 2) as f32;

    // First pass: compute h0 and omega
    for m in 0..n {
        for nn in 0..n {
            let kx = 2.0 * std::f32::consts::PI / l * (m as f32 - half_n);
            let kz = 2.0 * std::f32::consts::PI / l * (nn as f32 - half_n);
            let k_mag = (kx * kx + kz * kz).sqrt();

            let idx = m * n + nn;
            omega[idx] = (g * k_mag).sqrt();

            let spectrum = jonswap_spectrum(kx, kz, &config);
            let xi = box_muller(&mut rng);
            let scale = (spectrum.sqrt()) * std::f32::consts::FRAC_1_SQRT_2;
            h0[idx] = xi * scale;
        }
    }

    // Second pass: compute h0_conj from mirrored indices
    for m in 0..n {
        for nn in 0..n {
            let idx = m * n + nn;
            let mirror_m = (n - m) % n;
            let mirror_n = (n - nn) % n;
            let mirror_idx = mirror_m * n + mirror_n;
            h0_conj[idx] = h0[mirror_idx].conj();
        }
    }

    // Spawn ocean mesh entity
    let ocean_mesh = build_ocean_mesh(n, config.patch_length, MESH_STEP);
    commands.spawn((
        Mesh3d(meshes.add(ocean_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.3, 0.6),
            alpha_mode: AlphaMode::Opaque,
            double_sided: true,
            cull_mode: None,
            perceptual_roughness: 0.3,
            metallic: 0.1,
            ..default()
        })),
        NotShadowCaster,
        Transform::default(),
    )).insert(OceanMesh);

    commands.insert_resource(OceanSimulation {
        config,
        h0,
        h0_conj,
        omega,
        spectrum: vec![Complex32::new(0.0, 0.0); n2],
        heights: vec![0.0; n2],
        normals: vec![Vec3::Y; n2],
        row_scratch: vec![Complex32::new(0.0, 0.0); n],
    });
}

fn update_ocean(
    mut sim: ResMut<OceanSimulation>,
    time: Res<Time>,
    mut gizmos: Gizmos,
    ocean_query: Query<&Mesh3d, With<OceanMesh>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let t = time.elapsed_secs();
    let n = sim.config.grid_size;
    let n2 = n * n;
    let patch_length = sim.config.patch_length;

    // Step 1: Time-evolve spectrum
    for idx in 0..n2 {
        let a = sim.h0[idx];
        let b = sim.h0_conj[idx];
        let wt = sim.omega[idx] * t;
        let exp_iwt = Complex32::from_polar(1.0, wt);
        sim.spectrum[idx] = a * exp_iwt + b * exp_iwt.conj();
    }

    // Step 2: 2D IFFT
    {
        let sim = &mut *sim;
        let scratch = &mut sim.row_scratch;
        ifft_2d(&mut sim.spectrum, n, scratch);
        for idx in 0..n2 {
            sim.heights[idx] = sim.spectrum[idx].re;
        }
    }

    // Debug: print height range every ~60 frames
    if (time.elapsed_secs() * 10.0) as u32 % 10 == 0 {
        let min_h = sim.heights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_h = sim.heights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        info!("Ocean height range: {:.3} to {:.3}", min_h, max_h);
    }

    // Step 3: Compute normals
    let sim = &mut *sim;
    compute_normals(&sim.heights, &mut sim.normals, n, patch_length);

    // Step 4: Update mesh vertices (bilinear interpolation for fractional MESH_STEP)
    if let Ok(mesh_handle) = ocean_query.single() {
        if let Some(mesh) = meshes.get_mut(mesh_handle) {
            let mesh_n = (n as f32 / MESH_STEP) as usize;
            if let Some(positions) = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION) {
                if let bevy::mesh::VertexAttributeValues::Float32x3(pos) = positions {
                    for mz in 0..mesh_n {
                        for mx in 0..mesh_n {
                            let h = sample_bilinear(&sim.heights, n, mx as f32 * MESH_STEP, mz as f32 * MESH_STEP);
                            pos[mz * mesh_n + mx][1] = h;
                        }
                    }
                }
            }
            // Update normals from computed normals (interpolate each component)
            if let Some(norm_attr) = mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL) {
                if let bevy::mesh::VertexAttributeValues::Float32x3(norms) = norm_attr {
                    for mz in 0..mesh_n {
                        for mx in 0..mesh_n {
                            let fx = mx as f32 * MESH_STEP;
                            let fz = mz as f32 * MESH_STEP;
                            let src_normal = sample_bilinear_normal(&sim.normals, n, fx, fz);
                            norms[mz * mesh_n + mx] = src_normal.into();
                        }
                    }
                }
            }
        }
    }

    // // Step 5: Debug gizmo wireframe (sparse, every 8th point)
    // let gizmo_step = 8;
    // let dx = patch_length / n as f32;
    // let offset_x = -patch_length / 2.0;
    // let offset_z = -patch_length / 2.0;
    //
    // for z in (0..n - 1).step_by(gizmo_step) {
    //     for x in (0..n - 1).step_by(gizmo_step) {
    //         let h00 = sim.heights[z * n + x];
    //         let h10 = sim.heights[z * n + (x + gizmo_step).min(n - 1)];
    //         let h01 = sim.heights[((z + gizmo_step).min(n - 1)) * n + x];
    //
    //         let p00 = Vec3::new(offset_x + x as f32 * dx, h00, offset_z + z as f32 * dx);
    //         let p10 = Vec3::new(
    //             offset_x + (x + gizmo_step).min(n - 1) as f32 * dx,
    //             h10,
    //             offset_z + z as f32 * dx,
    //         );
    //         let p01 = Vec3::new(
    //             offset_x + x as f32 * dx,
    //             h01,
    //             offset_z + ((z + gizmo_step).min(n - 1)) as f32 * dx,
    //         );
    //
    //         gizmos.line(p00, p10, Color::srgb(0.2, 0.5, 1.0));
    //         gizmos.line(p00, p01, Color::srgb(0.2, 0.5, 1.0));
    //     }
    // }
}
