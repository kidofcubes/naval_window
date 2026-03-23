use bevy::app::{App, Plugin, Startup, Update};
use bevy::math::Vec2;
use bevy::prelude::*;
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
            patch_length: 200.0,
            wind_speed: 10.0,
            wind_dir: Vec2::new(1.0, 0.0).normalize(),
            amplitude: 1.0,
            gravity: 9.81,
            fetch: 100_000.0,
            gamma: 3.3,
        }
    }
}

#[derive(Resource)]
struct OceanSimulation {
    config: OceanConfig,
    h0: Vec<(f32, f32)>,
    h0_conj: Vec<(f32, f32)>,
    omega: Vec<f32>,
    spectrum_real: Vec<f32>,
    spectrum_imag: Vec<f32>,
    heights: Vec<f32>,
    normals: Vec<[f32; 3]>,
    row_scratch_re: Vec<f32>,
    row_scratch_im: Vec<f32>,
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

fn box_muller(rng: &mut impl Rng) -> (f32, f32) {
    let u1: f32 = rng.random::<f32>().max(1e-10);
    let u2: f32 = rng.random::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

fn ifft_2d(
    re: &mut [f32],
    im: &mut [f32],
    n: usize,
    scratch_re: &mut Vec<f32>,
    scratch_im: &mut Vec<f32>,
) {
    scratch_re.resize(n, 0.0);
    scratch_im.resize(n, 0.0);

    // Row pass
    for z in 0..n {
        let offset = z * n;
        scratch_re.copy_from_slice(&re[offset..offset + n]);
        scratch_im.copy_from_slice(&im[offset..offset + n]);

        phastft::fft_32(scratch_re, scratch_im, Direction::Reverse);

        re[offset..offset + n].copy_from_slice(scratch_re);
        im[offset..offset + n].copy_from_slice(scratch_im);
    }

    // Column pass
    for x in 0..n {
        for z in 0..n {
            scratch_re[z] = re[z * n + x];
            scratch_im[z] = im[z * n + x];
        }

        phastft::fft_32(scratch_re, scratch_im, Direction::Reverse);

        for z in 0..n {
            // fftshift sign correction: (-1)^(x+z)
            let sign = if (x + z) % 2 == 0 { 1.0 } else { -1.0 };
            re[z * n + x] = scratch_re[z] * sign;
        }
    }
}

fn compute_normals(heights: &[f32], normals: &mut [[f32; 3]], n: usize, patch_length: f32) {
    let dx = patch_length / n as f32;
    for z in 0..n {
        for x in 0..n {
            let left = heights[z * n + (x + n - 1) % n];
            let right = heights[z * n + (x + 1) % n];
            let down = heights[((z + n - 1) % n) * n + x];
            let up = heights[((z + 1) % n) * n + x];

            let dhdx = (right - left) / (2.0 * dx);
            let dhdz = (up - down) / (2.0 * dx);

            let normal = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
            normals[z * n + x] = [normal.x, normal.y, normal.z];
        }
    }
}

fn init_ocean(mut commands: Commands) {
    let config = OceanConfig::default();
    let n = config.grid_size;
    let n2 = n * n;

    let mut h0 = vec![(0.0f32, 0.0f32); n2];
    let mut h0_conj = vec![(0.0f32, 0.0f32); n2];
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
            let (xi_r, xi_i) = box_muller(&mut rng);
            let scale = (spectrum.sqrt()) * std::f32::consts::FRAC_1_SQRT_2;
            h0[idx] = (xi_r * scale, xi_i * scale);
        }
    }

    // Second pass: compute h0_conj from mirrored indices
    for m in 0..n {
        for nn in 0..n {
            let idx = m * n + nn;
            let mirror_m = (n - m) % n;
            let mirror_n = (n - nn) % n;
            let mirror_idx = mirror_m * n + mirror_n;
            // conjugate: negate imaginary part
            h0_conj[idx] = (h0[mirror_idx].0, -h0[mirror_idx].1);
        }
    }

    commands.insert_resource(OceanSimulation {
        config,
        h0,
        h0_conj,
        omega,
        spectrum_real: vec![0.0; n2],
        spectrum_imag: vec![0.0; n2],
        heights: vec![0.0; n2],
        normals: vec![[0.0, 1.0, 0.0]; n2],
        row_scratch_re: vec![0.0; n],
        row_scratch_im: vec![0.0; n],
    });
}

fn update_ocean(mut sim: ResMut<OceanSimulation>, time: Res<Time>, mut gizmos: Gizmos) {
    let t = time.elapsed_secs();
    let n = sim.config.grid_size;
    let n2 = n * n;
    let patch_length = sim.config.patch_length;

    // Step 1: Time-evolve spectrum
    for idx in 0..n2 {
        let (a_r, a_i) = sim.h0[idx];
        let (b_r, b_i) = sim.h0_conj[idx];
        let wt = sim.omega[idx] * t;
        let (sin_wt, cos_wt) = wt.sin_cos();

        sim.spectrum_real[idx] = a_r * cos_wt - a_i * sin_wt + b_r * cos_wt + b_i * sin_wt;
        sim.spectrum_imag[idx] = a_r * sin_wt + a_i * cos_wt - b_r * sin_wt + b_i * cos_wt;
    }

    // Step 2: 2D IFFT
    // Need to split borrows carefully
    let (spectrum_real, rest) = {
        let sim = &mut *sim;
        let re = &mut sim.spectrum_real;
        let im = &mut sim.spectrum_imag;
        let scratch_re = &mut sim.row_scratch_re;
        let scratch_im = &mut sim.row_scratch_im;
        ifft_2d(re, im, n, scratch_re, scratch_im);
        (&sim.spectrum_real as &[f32], &mut sim.heights)
    };

    // Copy real part to heights (already done by ifft_2d into spectrum_real with sign correction)
    rest.copy_from_slice(spectrum_real);

    // Step 3: Compute normals
    let sim = &mut *sim;
    compute_normals(&sim.heights, &mut sim.normals, n, patch_length);

    // Step 4: Debug visualization with gizmos
    // Draw a subset of the grid as lines (every 4th point to keep it reasonable)
    let step = 4;
    let dx = patch_length / n as f32;
    let offset_x = -patch_length / 2.0;
    let offset_z = -patch_length / 2.0;

    for z in (0..n - 1).step_by(step) {
        for x in (0..n - 1).step_by(step) {
            let h00 = sim.heights[z * n + x];
            let h10 = sim.heights[z * n + (x + step).min(n - 1)];
            let h01 = sim.heights[(z + step).min(n - 1) * n + x];

            let p00 = Vec3::new(offset_x + x as f32 * dx, h00, offset_z + z as f32 * dx);
            let p10 = Vec3::new(
                offset_x + (x + step).min(n - 1) as f32 * dx,
                h10,
                offset_z + z as f32 * dx,
            );
            let p01 = Vec3::new(
                offset_x + x as f32 * dx,
                h01,
                offset_z + (z + step).min(n - 1) as f32 * dx,
            );

            gizmos.line(p00, p10, Color::srgb(0.2, 0.5, 1.0));
            gizmos.line(p00, p01, Color::srgb(0.2, 0.5, 1.0));
        }
    }
}
