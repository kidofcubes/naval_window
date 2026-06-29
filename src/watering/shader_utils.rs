use std::f32::consts::{PI, TAU};
use bevy::prelude::*;
use bevy::render::render_resource::{
    encase, AddressMode, Extent3d, FilterMode, Sampler, SamplerDescriptor, ShaderType,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::RenderDevice;
use encase::internal::WriteInto;
use crate::big_watering::OceanConfig;
use crate::watering::{DIRECTIONS, GRID_SCALE, GRID_X_SIZE, GRID_Z_SIZE, MAX_ZETA, MIN_ZETA, PROFILE_BUFFER_LENGTH, TOTAL_ZETA_NUMBERS, WIND_SPEED};
use crate::watering::cpu_debug_impl::CpuDebugWateringResources;
// ─── Texture / sampler factories ────────────────────────────────────────────

/// Build a storage-capable 2D or 2D-array `Image` whose dimensionality is
/// inferred from `size.depth_or_array_layers`. Zero-initialised CPU data is
/// allocated so the asset is valid before the GPU upload.
pub fn make_storage_image(
    label: &'static str,
    size: Extent3d,
    format: TextureFormat,
) -> Image {
    let bytes_per_pixel = format
        .block_copy_size(None)
        .expect("storage textures must have a known copy size") as usize;
    let pixel_count = (size.width as usize)
        * (size.height as usize)
        * (size.depth_or_array_layers as usize);

    Image {
        texture_descriptor: TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        },
        data: Some(vec![0u8; pixel_count * bytes_per_pixel]),
        ..default()
    }
}

pub fn create_clamp_linear_sampler(device: &RenderDevice) -> Sampler {
    device.create_sampler(&SamplerDescriptor {
        label: Some("amplitude_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    })
}

// ─── Dispatch sizes ─────────────────────────────────────────────────────────

const GRID_WORKGROUP_XY: usize = 32;
const PROFILE_WORKGROUP_X: usize = 128;

pub fn grid_workgroups() -> (u32, u32, u32) {
    (
        GRID_X_SIZE.div_ceil(GRID_WORKGROUP_XY) as u32,
        GRID_Z_SIZE.div_ceil(GRID_WORKGROUP_XY) as u32,
        DIRECTIONS as u32,
    )
}

pub fn profile_workgroups() -> (u32, u32, u32) {
    (PROFILE_BUFFER_LENGTH.div_ceil(PROFILE_WORKGROUP_X) as u32, 1, 1)
}

// ─── Uniform encoding ───────────────────────────────────────────────────────

pub fn encode_uniform<T: ShaderType + WriteInto>(data: &T) -> Vec<u8> {
    let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
    buf.write(data).expect("ShaderType write failed");
    buf.into_inner()
}


pub fn precompute_jonswap_spectrum_amplitudes(
    min_zeta: f32, max_zeta: f32, dir_count: usize, zeta_count: usize, integration_steps: usize
) -> Vec<f32> {
    let zeta_full_range = max_zeta - min_zeta;
    let wind_dir_angle = (-0.40 * PI)+(((-2.0)/DIRECTIONS as f32)*TAU);
    let wind_dir = Vec2::new(f32::cos(wind_dir_angle), f32::sin(wind_dir_angle)).normalize();
    let ocean_config = OceanConfig {
        grid_size: GRID_X_SIZE,
        patch_length: GRID_SCALE*GRID_X_SIZE as f32,
        wind_speed: WIND_SPEED,
        // wind_dir: Vec2::new(1.0, 1.0).normalize(),
        wind_dir: wind_dir,
        // wind_dir: Vec2::new(-1.0, 1.0).normalize(),
        amplitude: 1000.0,
        gravity: 9.81,
        fetch: 1000_000.0,
        gamma: 3.3,
    };
    let mut jonswap_amplitudes = vec![0.0; zeta_count*dir_count];


    for theta_index in 0..dir_count {
        let theta = (theta_index as f32 + 0.0) * (TAU / dir_count as f32);

        let dir = vec3(f32::cos(theta), 0.0, f32::sin(theta));


        for zeta_index in 0..zeta_count {
            let min_zeta = min_zeta + (zeta_full_range * (((zeta_index as f32) - 0.0) / (zeta_count as f32)));
            let max_zeta = min_zeta + (zeta_full_range * (((zeta_index as f32) + 1.0) / (zeta_count as f32)));
            let mid_zeta = min_zeta + (zeta_full_range * (((zeta_index as f32) + 0.5) / (zeta_count as f32)));

            let result = integrate(min_zeta, max_zeta, |zeta| {
                let wave_length = 2.0f32.powf(zeta);
                let wave_number = TAU / wave_length;

                let jonswap_value = crate::big_watering::jonswap_spectrum(wave_number * dir.x, wave_number * dir.z, &ocean_config);
                return jonswap_value;
            }, integration_steps as i32);

            let result = f32::sqrt(result * (TAU/(DIRECTIONS as f32)));
            jonswap_amplitudes[(zeta_index * dir_count) + theta_index] = result;
        }
    }
    return jonswap_amplitudes;
}


// ─── Offline wave-spectrum precompute (was `pre_compute.rs`) ────────────────

pub fn compute_group_speeds(min_zeta: f32, max_zeta: f32, wind_speed: f32, zeta_count: usize) -> Vec<f32> {
    let d_zeta = (max_zeta - min_zeta) / (zeta_count as f32);
    let group_speeds = (0..zeta_count).map(|i| {
        let bin_min = min_zeta + (d_zeta * (i as f32 - 0.0));
        let bin_max = bin_min + (d_zeta * (i as f32 + 1.0));
        compute_group_speed(bin_min, bin_max, wind_speed)
    }).collect::<Vec<_>>();
    println!("GROUP SPEEDS ARE {:?}", group_speeds);
    group_speeds
}

pub fn compute_group_speed(min_zeta: f32, max_zeta: f32, wind_speed: f32) -> f32 {
    let result = integrate(min_zeta, max_zeta, |zeta| {
        let wave_length = 2.0f32.powf(zeta);
        let wave_number = TAU / wave_length;
        let cg = 0.5 * f32::sqrt(crate::watering::GRAVITY / wave_number);
        let density = wave_energy_density(zeta, wind_speed);
        vec2(cg * density, density) / 100.0
    }, 100);
    3.0 * result.x / result.y
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
    result
}

/// Pierson-Moskowitz wave energy density.
pub fn wave_energy_density(zeta: f32, wind_speed: f32) -> f32 {
    let a = 1.1f32.powf(1.5f32 * zeta);
    let b = f32::exp(-1.8038897788076411 * 4f32.powf(zeta) / wind_speed.powf(4f32));
    0.139098 * f32::sqrt(a * b)
}
