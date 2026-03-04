use std::f32::consts::TAU;
use bevy::prelude::*;

fn compute_group_speeds(min_zeta: f32, max_zeta: f32, wind_speed: f32, zeta_count: usize) -> Vec<f32> {
    let d_zeta = (max_zeta - min_zeta) / (zeta_count as f32);
    let group_speeds = (0..zeta_count).map(|i| {
        let min_zeta = min_zeta + (d_zeta * (i as f32 - 0.5));
        let max_zeta = min_zeta + (d_zeta * (i as f32 + 0.5));
        return compute_group_speed(min_zeta, max_zeta, wind_speed);
    }).collect::<Vec<_>>();
    return group_speeds;
}

fn compute_group_speed(min_zeta: f32, max_zeta: f32, wind_speed: f32) -> f32{
    let result = integrate(min_zeta, max_zeta, |zeta| {
        let wave_length = 2.0f32.powf(zeta);
        let wave_number = TAU / wave_length;
        let cg = 0.5 * f32::sqrt(crate::watering::GRAVITY / wave_number);
        let density = wave_energy_density(zeta, wind_speed);

        return vec2(cg * density, density) / 100.0;
    }, 100);
    return 3.0 * result.x / result.y;
}

fn integrate<T, F>(min: f32, max: f32, f: F, n: i32) -> T
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
fn wave_energy_density(zeta: f32, wind_speed: f32) -> f32 {
    let a = 1.1f32.powf(1.5f32 * zeta);
    let b = f32::exp(-1.8038897788076411 * 4f32.powf(zeta) / wind_speed.powf(4f32));
    return 0.139098 * f32::sqrt(a * b);
}