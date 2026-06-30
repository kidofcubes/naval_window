mod compute_shader;

use std::collections::HashMap;
use bevy::app::{App, Plugin, Startup, Update};
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{Extent3d, ShaderType, TextureFormat};
use bevy::render::{Render, RenderApp, RenderSystems};
use crate::watering::shader_utils::make_storage_image;

pub const FFT_GRID_SIZE: usize = 512;
pub const FFT_CASCADE_COUNT: usize = 4;
pub const FFT_SPECTRUM_COUNT: usize = 5;

/// (patch_length_m, k_min, k_max, _pad). One per cascade.
///
/// Patches are non-power-of-2 / coprime-ish so their periods don't sync
/// (LCM is in the millions of meters, way past visible range). Each cascade
/// owns wavenumbers from 2π / its-own-patch up to 2π / next-smaller-patch,
/// which is the natural Nyquist-like boundary that keeps any single k-bin
/// from being represented by more than one cascade.
///   997 m → k ∈ [0,         2π/251 ≈ 0.025]   long ocean swell
///   251 m → k ∈ [2π/251,    2π/53  ≈ 0.119]   ship-scale waves (JONSWAP peak ~ 0.10)
///    53 m → k ∈ [2π/53,     2π/13  ≈ 0.483]   wind chop
///    13 m → k ∈ [2π/13,     ∞              ]   ripples
pub const FFT_CASCADES: [Vec4; FFT_CASCADE_COUNT] = [
    Vec4::new(37.0, 0.0,   0.025,  0.0),
    Vec4::new(1021.0, 0.025,  0.119,  0.0),
    Vec4::new( 571.0, 0.119,  0.483,  0.0),
    Vec4::new( 127.0, 0.119,  0.483,  0.0),
    // Vec4::new( 127.0, 0.483,  100.0,  0.0),
];

pub struct FftWavesPlugin;

impl Plugin for FftWavesPlugin {
    fn build(&self, app: &mut App) {
        app
            .insert_resource(FftSettings { elapsed_time: 0.0, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0 })
            .insert_resource(FftCascadesUniform { cascades: FFT_CASCADES })
            .insert_resource(FftTextures::default())
            .add_systems(Startup, setup_textures)
            .add_systems(Update, tick_time)
            .add_plugins(ExtractResourcePlugin::<FftSettings>::default())
            .add_plugins(ExtractResourcePlugin::<FftCascadesUniform>::default())
            .add_plugins(ExtractResourcePlugin::<FftTextures>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            (
                compute_shader::prepare_gpu_resources.in_set(RenderSystems::PrepareResources),
                compute_shader::queue_bind_groups.in_set(RenderSystems::Queue),
            ),
        );
        render_app.add_systems(
            Core3d,
            compute_shader::render_pass.in_set(Core3dSystems::Prepass),
        );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<compute_shader::FftPipelines>();
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum FftTexture {
    H0,
    H0Conj,
    Omega,
    SpectraA,
    SpectraB,
    Displacement,
    Slope,
}

#[derive(Resource, ExtractResource, Clone, Default)]
pub struct FftTextures {
    pub map: HashMap<FftTexture, Handle<Image>>,
}

impl FftTextures {
    pub fn get(&self, key: FftTexture) -> Option<&Handle<Image>> { self.map.get(&key) }
    pub fn insert(&mut self, key: FftTexture, handle: Handle<Image>) { self.map.insert(key, handle); }
}

#[derive(Resource, ExtractResource, Clone, ShaderType)]
pub struct FftSettings {
    pub elapsed_time: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

#[derive(Resource, ExtractResource, Clone, ShaderType)]
pub struct FftCascadesUniform {
    pub cascades: [Vec4; FFT_CASCADE_COUNT],
}

pub fn setup_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let cascade_size = Extent3d {
        width: FFT_GRID_SIZE as u32,
        height: FFT_GRID_SIZE as u32,
        depth_or_array_layers: FFT_CASCADE_COUNT as u32,
    };
    let spectra_size = Extent3d {
        width: FFT_GRID_SIZE as u32,
        height: FFT_GRID_SIZE as u32,
        depth_or_array_layers: (FFT_SPECTRUM_COUNT * FFT_CASCADE_COUNT) as u32,
    };

    let mut textures = FftTextures::default();
    textures.insert(
        FftTexture::H0,
        images.add(make_storage_image("fft_h0", cascade_size, TextureFormat::Rg32Float)),
    );
    textures.insert(
        FftTexture::H0Conj,
        images.add(make_storage_image("fft_h0_conj", cascade_size, TextureFormat::Rg32Float)),
    );
    textures.insert(
        FftTexture::Omega,
        images.add(make_storage_image("fft_omega", cascade_size, TextureFormat::R32Float)),
    );
    textures.insert(
        FftTexture::SpectraA,
        images.add(make_storage_image("fft_spectra_a", spectra_size, TextureFormat::Rg32Float)),
    );
    textures.insert(
        FftTexture::SpectraB,
        images.add(make_storage_image("fft_spectra_b", spectra_size, TextureFormat::Rg32Float)),
    );
    textures.insert(
        FftTexture::Displacement,
        images.add(make_storage_image("fft_displacement", cascade_size, TextureFormat::Rgba32Float)),
    );
    textures.insert(
        FftTexture::Slope,
        images.add(make_storage_image("fft_slope", cascade_size, TextureFormat::Rgba32Float)),
    );

    commands.insert_resource(textures);
}

fn tick_time(time: Res<Time>, mut settings: ResMut<FftSettings>) {
    settings.elapsed_time = time.elapsed_secs();
}
