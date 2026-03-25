use std::f32::consts::{PI, TAU};
use bevy::prelude::*;
use num_complex::Complex32;
use rand::Rng;

pub fn compute_group_speeds(min_zeta: f32, max_zeta: f32, wind_speed: f32, zeta_count: usize) -> Vec<f32> {
    let d_zeta = (max_zeta - min_zeta) / (zeta_count as f32);
    let group_speeds = (0..zeta_count).map(|i| {
        let min_zeta = min_zeta + (d_zeta * (i as f32 - 0.0));
        let max_zeta = min_zeta + (d_zeta * (i as f32 + 1.0));
        return compute_group_speed(min_zeta, max_zeta, wind_speed);
    }).collect::<Vec<_>>();
    println!("GROUP SPEEDS ARE {:?}", group_speeds);
    return group_speeds;
}

pub fn compute_group_speed(min_zeta: f32, max_zeta: f32, wind_speed: f32) -> f32{
    let result = integrate(min_zeta, max_zeta, |zeta| {
        let wave_length = 2.0f32.powf(zeta);
        let wave_number = TAU / wave_length;
        let cg = 0.5 * f32::sqrt(crate::watering::GRAVITY / wave_number);
        let density = wave_energy_density(zeta, wind_speed);

        return vec2(cg * density, density) / 100.0;
    }, 100);
    return 3.0 * result.x / result.y;
}

pub fn integrate<T, F>(min: f32, max: f32, f: F, n: i32) -> T
where
    F: Fn(f32) -> T,
    T: std::ops::Add<Output = T> + std::ops::Mul<f32, Output = T>,
{
    let dx = (max - min) / n as f32;
    let mut x = min + (0.5f32 * dx);
    let mut result = f(x);
    for _ in 1..n {
        x += dx;
        result = result + (f(x) * dx);
    }
    return result;
}

/// so called Pierson-Moskowitz wave energy density function
pub fn wave_energy_density(zeta: f32, wind_speed: f32) -> f32 {
    let a = 1.1f32.powf(1.5f32 * zeta);
    let b = f32::exp(-1.8038897788076411 * 4f32.powf(zeta) / wind_speed.powf(4f32));
    return 0.139098 * f32::sqrt(a * b);
}

// ─── FFT Ocean Precomputation ────────────────────────────────────────────────

pub struct FftOceanConfig {
    pub grid_size: usize,
    pub patch_length: f32,
    pub wind_speed: f32,
    pub wind_dir: Vec2,
    pub amplitude: f32,
    pub gravity: f32,
    pub fetch: f32,
    pub gamma: f32,
}

impl Default for FftOceanConfig {
    fn default() -> Self {
        Self {
            grid_size: 256,
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

fn jonswap_spectrum(kx: f32, kz: f32, config: &FftOceanConfig) -> f32 {
    let wave_number = (kx * kx + kz * kz).sqrt();
    if wave_number < 1e-8 {
        return 0.0;
    }

    let gravity = config.gravity;
    let angular_frequency = (gravity * wave_number).sqrt();

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

    // Convert to k-space
    let d_omega_dk = gravity / (2.0 * angular_frequency);
    let spectrum_k = jonswap * d_omega_dk / (2.0 * PI);

    // Directional spreading: cos^2(theta)
    let wave_dir = Vec2::new(kx, kz) / wave_number;
    let cos_theta = wave_dir.dot(config.wind_dir);
    let directional = if cos_theta > 0.0 { cos_theta * cos_theta } else { 0.0 };

    (config.amplitude * spectrum_k * directional).max(0.0)
}

fn box_muller(rng: &mut impl Rng) -> Complex32 {
    let u1: f32 = rng.random::<f32>().max(1e-10);
    let u2: f32 = rng.random::<f32>();
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = TAU * u2;
    Complex32::new(radius * angle.cos(), radius * angle.sin())
}

/// Returns (initial_spectrum_bytes, initial_spectrum_conj_bytes, angular_frequency_bytes)
/// for 256x256 textures. Spectrum textures are Rg32Float (2 floats per texel),
/// angular frequency is R32Float (1 float per texel).
pub fn generate_fft_initial_data(config: &FftOceanConfig) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let grid_size = config.grid_size;
    let patch_length = config.patch_length;
    let gravity = config.gravity;
    let half_grid = (grid_size / 2) as f32;

    let zero = Complex32::new(0.0, 0.0);
    let mut initial_spectrum = vec![vec![zero; grid_size]; grid_size];
    let mut angular_frequency = vec![vec![0.0f32; grid_size]; grid_size];

    let mut rng = rand::rng();

    for row in 0..grid_size {
        for col in 0..grid_size {
            let kx = TAU * ((row as f32 - half_grid) / patch_length);
            let kz = TAU * ((col as f32 - half_grid) / patch_length);
            let wave_number = Vec2::new(kx, kz).length();

            angular_frequency[row][col] = (gravity * wave_number).sqrt();

            let spectrum_value = jonswap_spectrum(kx, kz, config);
            let gaussian_noise = box_muller(&mut rng);
            let scale = spectrum_value.sqrt() * std::f32::consts::FRAC_1_SQRT_2;
            initial_spectrum[row][col] = gaussian_noise * scale;
        }
    }

    // Conjugate from mirrored indices
    let mut initial_spectrum_conj = vec![vec![zero; grid_size]; grid_size];
    for row in 0..grid_size {
        for col in 0..grid_size {
            let mirror_row = (grid_size - row) % grid_size;
            let mirror_col = (grid_size - col) % grid_size;
            initial_spectrum_conj[row][col] = initial_spectrum[mirror_row][mirror_col].conj();
        }
    }

    // Pack into byte arrays
    // Spectrum: Rg32Float = 8 bytes per texel (re, im)
    let mut spectrum_bytes = vec![0u8; grid_size * grid_size * 8];
    let mut conj_bytes = vec![0u8; grid_size * grid_size * 8];
    let mut freq_bytes = vec![0u8; grid_size * grid_size * 4];

    for row in 0..grid_size {
        for col in 0..grid_size {
            let idx = row * grid_size + col;

            let s = initial_spectrum[row][col];
            spectrum_bytes[idx * 8..idx * 8 + 4].copy_from_slice(&s.re.to_le_bytes());
            spectrum_bytes[idx * 8 + 4..idx * 8 + 8].copy_from_slice(&s.im.to_le_bytes());

            let c = initial_spectrum_conj[row][col];
            conj_bytes[idx * 8..idx * 8 + 4].copy_from_slice(&c.re.to_le_bytes());
            conj_bytes[idx * 8 + 4..idx * 8 + 8].copy_from_slice(&c.im.to_le_bytes());

            let w = angular_frequency[row][col];
            freq_bytes[idx * 4..idx * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
    }

    (spectrum_bytes, conj_bytes, freq_bytes)
}