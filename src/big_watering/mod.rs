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
    initial_spectrum: Vec<Vec<Complex32>>,
    initial_spectrum_conjugate: Vec<Vec<Complex32>>,
    angular_frequency: Vec<Vec<f32>>,
    evolved_spectrum: Vec<Vec<Complex32>>,
    heights: Vec<Vec<f32>>,
    normals: Vec<Vec<Vec3>>,
    fft_scratch: Vec<Complex32>,
}

fn jonswap_spectrum(kx: f32, kz: f32, config: &OceanConfig) -> f32 {
    let wave_number = (kx * kx + kz * kz).sqrt();
    if wave_number < 1e-8 {
        return 0.0;
    }

    let gravity = config.gravity;
    let angular_frequency = (gravity * wave_number).sqrt();

    // Peak frequency (JONSWAP)
    let omega_peak = 22.0 * (gravity * gravity / (config.wind_speed * config.fetch)).powf(1.0 / 3.0);

    if angular_frequency < 1e-8 {
        return 0.0;
    }

    // Pierson-Moskowitz base spectrum
    let alpha = 0.0081;
    let omega_ratio = omega_peak / angular_frequency;
    let pierson_moskowitz = (alpha * gravity * gravity / angular_frequency.powi(5))
        * (-1.25 * omega_ratio.powi(4)).exp();

    // JONSWAP peak enhancement
    let sigma = if angular_frequency <= omega_peak { 0.07 } else { 0.09 };
    let exponent = ((angular_frequency - omega_peak).powi(2)) / (2.0 * sigma * sigma * omega_peak * omega_peak);
    let gamma_factor = config.gamma.powf((-exponent).exp());

    let jonswap = pierson_moskowitz * gamma_factor;

    // Convert to k-space: S(k) = S(omega) * d_omega/dk / (2*pi)
    // d_omega/dk = g / (2 * omega) for deep water dispersion
    let d_omega_dk = gravity / (2.0 * angular_frequency);
    let spectrum_k = jonswap * d_omega_dk / (2.0 * std::f32::consts::PI);

    // Directional spreading: cos^2(theta) where theta is angle between k and wind
    let wave_dir = Vec2::new(kx, kz) / wave_number;
    let cos_theta = wave_dir.dot(config.wind_dir);
    let directional = if cos_theta > 0.0 {
        cos_theta * cos_theta
    } else {
        0.0
    };

    (config.amplitude * spectrum_k * directional).max(0.0)
}

fn box_muller(rng: &mut impl Rng) -> Complex32 {
    let u1: f32 = rng.random::<f32>().max(1e-10);
    let u2: f32 = rng.random::<f32>();
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = 2.0 * std::f32::consts::PI * u2;
    Complex32::new(radius * angle.cos(), radius * angle.sin())
}

fn ifft_2d(
    grid: &mut Vec<Vec<Complex32>>,
    scratch: &mut Vec<Complex32>,
) {
    let grid_size = grid.len();
    scratch.resize(grid_size, Complex32::new(0.0, 0.0));

    // Row pass
    for row in grid.iter_mut() {
        scratch.copy_from_slice(row);
        phastft::fft_32_interleaved(scratch, Direction::Reverse);
        row.copy_from_slice(scratch);
    }

    // Column pass
    for col in 0..grid_size {
        for row in 0..grid_size {
            scratch[row] = grid[row][col];
        }

        phastft::fft_32_interleaved(scratch, Direction::Reverse);

        for row in 0..grid_size {
            // fftshift sign correction: (-1)^(col+row)
            // Also multiply by N² to undo phastft's 1/N normalization per pass,
            // since Tessendorf expects an unnormalized DFT sum.
            let sign = if (col + row) % 2 == 0 { 1.0 } else { -1.0 };
            grid[row][col].re = scratch[row].re * sign * (grid_size * grid_size) as f32;
        }
    }
}

fn compute_normals(heights: &Vec<Vec<f32>>, normals: &mut Vec<Vec<Vec3>>, patch_length: f32) {
    let grid_size = heights.len();
    let cell_size = patch_length / grid_size as f32;
    for row in 0..grid_size {
        for col in 0..grid_size {
            let left = heights[row][(col + grid_size - 1) % grid_size];
            let right = heights[row][(col + 1) % grid_size];
            let down = heights[(row + grid_size - 1) % grid_size][col];
            let up = heights[(row + 1) % grid_size][col];

            let dh_dx = (right - left) / (2.0 * cell_size);
            let dh_dz = (up - down) / (2.0 * cell_size);

            normals[row][col] = Vec3::new(-dh_dx, 1.0, -dh_dz).normalize();
        }
    }
}

/// Bilinear interpolation into an N×N grid of f32.
fn sample_bilinear(grid: &Vec<Vec<f32>>, fractional_col: f32, fractional_row: f32) -> f32 {
    let grid_size = grid.len();
    let col0 = (fractional_col as usize).min(grid_size - 1);
    let row0 = (fractional_row as usize).min(grid_size - 1);
    let col1 = (col0 + 1).min(grid_size - 1);
    let row1 = (row0 + 1).min(grid_size - 1);
    let t_col = fractional_col - col0 as f32;
    let t_row = fractional_row - row0 as f32;

    let v00 = grid[row0][col0];
    let v10 = grid[row0][col1];
    let v01 = grid[row1][col0];
    let v11 = grid[row1][col1];

    let top = v00 + (v10 - v00) * t_col;
    let bot = v01 + (v11 - v01) * t_col;
    top + (bot - top) * t_row
}

/// Bilinear interpolation into an N×N grid of Vec3, returns normalized result.
fn sample_bilinear_normal(grid: &Vec<Vec<Vec3>>, fractional_col: f32, fractional_row: f32) -> Vec3 {
    let grid_size = grid.len();
    let col0 = (fractional_col as usize).min(grid_size - 1);
    let row0 = (fractional_row as usize).min(grid_size - 1);
    let col1 = (col0 + 1).min(grid_size - 1);
    let row1 = (row0 + 1).min(grid_size - 1);
    let t_col = fractional_col - col0 as f32;
    let t_row = fractional_row - row0 as f32;

    let v00 = grid[row0][col0];
    let v10 = grid[row0][col1];
    let v01 = grid[row1][col0];
    let v11 = grid[row1][col1];
    let top = v00 + (v10 - v00) * t_col;
    let bot = v01 + (v11 - v01) * t_col;
    let result = top + (bot - top) * t_row;
    result.normalize_or_zero()
}

fn build_ocean_mesh(grid_size: usize, patch_length: f32, mesh_step: f32) -> Mesh {
    let vertices_per_side = (grid_size as f32 / mesh_step) as usize;
    let cell_size = patch_length / grid_size as f32;
    let origin_offset = -patch_length / 2.0;

    let vertex_count = vertices_per_side * vertices_per_side;
    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(vertex_count);

    for row in 0..vertices_per_side {
        for col in 0..vertices_per_side {
            let world_x = origin_offset + col as f32 * mesh_step * cell_size;
            let world_z = origin_offset + row as f32 * mesh_step * cell_size;
            positions.push([world_x, 0.0, world_z]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([col as f32 / vertices_per_side as f32, row as f32 / vertices_per_side as f32]);
        }
    }

    let mut indices = Vec::with_capacity((vertices_per_side - 1) * (vertices_per_side - 1) * 6);
    for row in 0..(vertices_per_side - 1) {
        for col in 0..(vertices_per_side - 1) {
            let top_left = (row * vertices_per_side + col) as u32;
            let top_right = top_left + 1;
            let bottom_left = top_left + vertices_per_side as u32;
            let bottom_right = bottom_left + 1;
            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);
            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
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
    let grid_size = config.grid_size;

    let zero = Complex32::new(0.0, 0.0);
    let mut initial_spectrum = vec![vec![zero; grid_size]; grid_size];
    let mut angular_frequency = vec![vec![0.0f32; grid_size]; grid_size];

    let mut rng = rand::rng();
    let patch_length = config.patch_length;
    let gravity = config.gravity;
    let half_grid = (grid_size / 2) as f32;

    // First pass: compute initial_spectrum and angular_frequency
    for row in 0..grid_size {
        for col in 0..grid_size {
            let kx = 2.0 * std::f32::consts::PI * ((row as f32 - half_grid) / patch_length);
            let kz = 2.0 * std::f32::consts::PI * ((col as f32 - half_grid) / patch_length);

            //wave number is 1/wave length
            let wave_number = vec2(kx, kz).length();

            angular_frequency[row][col] = (gravity * wave_number).sqrt();

            let spectrum_value = jonswap_spectrum(kx, kz, &config);
            let gaussian_noise = box_muller(&mut rng);
            let scale = spectrum_value.sqrt() * std::f32::consts::FRAC_1_SQRT_2;
            initial_spectrum[row][col] = gaussian_noise * scale;
        }
    }

    // Second pass: compute conjugate from mirrored indices
    let mut initial_spectrum_conjugate = vec![vec![zero; grid_size]; grid_size];
    for row in 0..grid_size {
        for col in 0..grid_size {
            let mirror_row = (grid_size - row) % grid_size;
            let mirror_col = (grid_size - col) % grid_size;
            initial_spectrum_conjugate[row][col] = initial_spectrum[mirror_row][mirror_col].conj();
        }
    }

    // Spawn ocean mesh entity
    let ocean_mesh = build_ocean_mesh(grid_size, config.patch_length, MESH_STEP);
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
        initial_spectrum,
        initial_spectrum_conjugate,
        angular_frequency,
        evolved_spectrum: vec![vec![zero; grid_size]; grid_size],
        heights: vec![vec![0.0; grid_size]; grid_size],
        normals: vec![vec![Vec3::Y; grid_size]; grid_size],
        fft_scratch: vec![zero; grid_size],
    });
}

fn update_ocean(
    mut sim: ResMut<OceanSimulation>,
    time: Res<Time>,
    mut gizmos: Gizmos,
    ocean_query: Query<&Mesh3d, With<OceanMesh>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let elapsed = time.elapsed_secs();
    let grid_size = sim.config.grid_size;
    let patch_length = sim.config.patch_length;

    // Step 1: Time-evolve spectrum
    for row in 0..grid_size {
        for col in 0..grid_size {
            let base = sim.initial_spectrum[row][col];
            let conjugate = sim.initial_spectrum_conjugate[row][col];
            let phase = sim.angular_frequency[row][col] * elapsed;
            let exp_phase = Complex32::from_polar(1.0, phase);
            sim.evolved_spectrum[row][col] = base * exp_phase + conjugate * exp_phase.conj();
        }
    }

    // Step 2: 2D IFFT
    {
        let sim = &mut *sim;
        ifft_2d(&mut sim.evolved_spectrum, &mut sim.fft_scratch);
        for row in 0..grid_size {
            for col in 0..grid_size {
                sim.heights[row][col] = sim.evolved_spectrum[row][col].re;
            }
        }
    }

    // Debug: print height range every ~60 frames
    if (elapsed * 10.0) as u32 % 10 == 0 {
        let min_height = sim.heights.iter().flat_map(|row| row.iter()).cloned().fold(f32::INFINITY, f32::min);
        let max_height = sim.heights.iter().flat_map(|row| row.iter()).cloned().fold(f32::NEG_INFINITY, f32::max);
        info!("Ocean height range: {:.3} to {:.3}", min_height, max_height);
    }

    // Step 3: Compute normals
    let sim = &mut *sim;
    compute_normals(&sim.heights, &mut sim.normals, patch_length);

    // Step 4: Update mesh vertices (bilinear interpolation for fractional MESH_STEP)
    if let Ok(mesh_handle) = ocean_query.single() {
        if let Some(mesh) = meshes.get_mut(mesh_handle) {
            let vertices_per_side = (grid_size as f32 / MESH_STEP) as usize;
            if let Some(positions) = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION) {
                if let bevy::mesh::VertexAttributeValues::Float32x3(pos) = positions {
                    for mesh_row in 0..vertices_per_side {
                        for mesh_col in 0..vertices_per_side {
                            let height = sample_bilinear(&sim.heights, mesh_col as f32 * MESH_STEP, mesh_row as f32 * MESH_STEP);
                            pos[mesh_row * vertices_per_side + mesh_col][1] = height;
                        }
                    }
                }
            }
            if let Some(normal_attr) = mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL) {
                if let bevy::mesh::VertexAttributeValues::Float32x3(mesh_normals) = normal_attr {
                    for mesh_row in 0..vertices_per_side {
                        for mesh_col in 0..vertices_per_side {
                            let fractional_col = mesh_col as f32 * MESH_STEP;
                            let fractional_row = mesh_row as f32 * MESH_STEP;
                            let normal = sample_bilinear_normal(&sim.normals, fractional_col, fractional_row);
                            mesh_normals[mesh_row * vertices_per_side + mesh_col] = normal.into();
                        }
                    }
                }
            }
        }
    }
}
