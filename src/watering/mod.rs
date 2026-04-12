mod compute_shader;
mod pre_compute;
mod cpu_debug_impl;
pub(crate) mod water_lod;

use std::f32::consts::TAU;
use std::time::Instant;
use bevy::app::{App, Plugin, Startup, Update};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::prelude::*;
use bevy::prelude::ops::atan2;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::renderer::RenderDevice;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use bevy::render::render_resource::{Extent3d, ShaderType, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};
use crate::BuildData;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::render::render_graph::RenderGraphExt;
use crate::watering::cpu_debug_impl::CpuDebugWatering;
use crate::watering::pre_compute::{compute_group_speeds, generate_fft_initial_data};
use crate::watering::water_lod::WaterLodPlugin;

pub struct Watering;
impl Plugin for Watering{
    fn build(&self, app: &mut App) {
        app
            .insert_resource(WaterSettings {
                height_scale: 1.0,
                time_scale: 3.0,
                delta_time: 0.0,
                elapsed_time: 0.0,
            })
            .insert_resource(WaterDebugSettings {
                spawn_center_x: GRID_X_SIZE as f32 / 2.0,
                spawn_center_z: GRID_Z_SIZE as f32 / 2.0,
                radius: 5.0,
                scale: 5.0,
                inject: 0.0,
                _padding_a: 0.0,
                _padding_b: 0.0,
                _padding_c: 0.0,
            })
            .insert_resource(VoxelSettings {
                water_to_voxel: Mat4::IDENTITY,
                voxel_size: 0.0,
                grid_dims: Default::default(),
            })
            .insert_resource(VoxelGridHandle {
                voxel: None,
                sdf: None,
            })
            .insert_resource(WaterWaveGroupSpeeds {
                wave_group_speeds: compute_group_speeds(MIN_ZETA, MAX_ZETA, WIND_SPEED, TOTAL_ZETA_NUMBERS).iter().map(|x| vec4(*x * GRID_SCALE, 0.0, 0.0, 0.0)).collect::<Vec<_>>().try_into().unwrap()
            })
            .insert_resource(FftOceanConfig {
                grid_size: 256,
                patch_lengths: FFT_PATCH_LENGTHS,
                wind_speed: 10.0,
                wind_dir: Vec2::new(1.0, 0.0).normalize(),
                amplitude: 0.1,
                gravity: 9.81,
                fetch: 100_000.0,
                gamma: 3.3,
            })
            .insert_resource(FftOceanSettingsArray {
                cascades: std::array::from_fn(|i| FftOceanSettings {
                    elapsed_time: 0.0,
                    patch_length: FFT_PATCH_LENGTHS[i],
                    _padding_b: 0.0,
                    _padding_c: 0.0,
                }),
            })
            .insert_resource(VoxelSettings::default())
            .add_systems(Startup, (setup_textures))
            .add_systems(Update, debug_input_system)
            .add_systems(Update, (tick_water_time, sync_voxel_grid))
        // Advance simulation time each frame
        // .add_systems(Update, (tick_water_time, sync_ship_position))
        ;

        if(true) {
            app
                .add_plugins(ExtractResourcePlugin::<WaterSettings>::default())
                .add_plugins(ExtractResourcePlugin::<HeightFieldHandle>::default())
                .add_plugins(ExtractResourcePlugin::<WaterDebugSettings>::default())
                .add_plugins(ExtractResourcePlugin::<VoxelSettings>::default())
                .add_plugins(ExtractResourcePlugin::<VoxelGridHandle>::default())
                .add_plugins(ExtractResourcePlugin::<WaterWaveGroupSpeeds>::default())
                .add_plugins(ExtractResourcePlugin::<WaterSimulationHandles>::default())
                .add_plugins(ExtractResourcePlugin::<FftOceanSettingsArray>::default())
                .add_plugins(ExtractResourcePlugin::<FftOceanHandles>::default())
            ;
            let render_app = app.sub_app_mut(RenderApp);
            render_app
                // .init_resource::<WaterSettings>()
                .add_systems(
                    Render,
                    (
                        compute_shader::prepare_gpu_resources
                            .in_set(RenderSystems::PrepareResources),
                        // pipeline::upload_obstacle_mask
                        //     .in_set(RenderSystems::PrepareResources)
                        //     .after(resources::prepare_gpu_resources),
                        compute_shader::queue_bind_groups
                            .in_set(RenderSystems::Queue),
                    ),
                );

            // use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
            // use bevy::render::render_graph::RenderGraphApp;
            render_app
                .add_render_graph_node::<compute_shader::WaterSimulationNode>(
                    Core3d,
                    compute_shader::WaterNodeLabel,
                )
                // Run our compute passes before the main 3D pass
                .add_render_graph_edge(Core3d, compute_shader::WaterNodeLabel, Node3d::EarlyPrepass);
        }else{
            app.add_plugins(CpuDebugWatering);
        }
    }

    fn finish(&self, app: &mut App) {
        // Pipeline creation needs the RenderDevice, which is only available
        // after the renderer is initialised (in `finish`).
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<compute_shader::WaterPipelines>();
    }
}


pub fn setup_textures(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    fft_ocean_config: Res<FftOceanConfig>,
    // mut water_material: ResMut<Assets<water_stuff::ocean_mesh::WaterMaterial>>,
    // mut hf_handle: ResMut<HeightFieldHandle>,
    // mut commands: Commands,
){
    let offsets_image = Image {
        texture_descriptor: TextureDescriptor {
            label:               Some("height_field_mat"),
            size:                Extent3d { width: RENDER_X_SIZE as u32, height: RENDER_Z_SIZE as u32, depth_or_array_layers: 1 },
            mip_level_count:     1,
            sample_count:        1,
            dimension:           TextureDimension::D2,
            format:              TextureFormat::Rgba32Float,
            usage:               TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            view_formats:        &[],
        },
        data: Some(vec![0u8; (RENDER_X_SIZE * RENDER_Z_SIZE * 16) as usize]),
        ..default()
    };

    // Normal field texture (rgba32float for xyz normal + unused alpha)
    let normal_image = Image {
        texture_descriptor: TextureDescriptor {
            label:               Some("normal_field_mat"),
            size:                Extent3d { width: RENDER_X_SIZE as u32, height: RENDER_Z_SIZE as u32, depth_or_array_layers: 1 },
            mip_level_count:     1,
            sample_count:        1,
            dimension:           TextureDimension::D2,
            format:              TextureFormat::Rgba32Float,
            usage:               TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            view_formats:        &[],
        },
        data: Some(vec![0u8; (RENDER_X_SIZE * RENDER_Z_SIZE * 16) as usize]), // 16 bytes per pixel (4 x f32)
        ..default()
    };

    // Create placeholder amplitude texture
    let amplitude_image_a = Image {
        texture_descriptor: TextureDescriptor {
            label:               Some("amplitude_texture"),
            size:                Extent3d { 
                width: GRID_X_SIZE as u32, 
                height: GRID_Z_SIZE as u32, 
                depth_or_array_layers: (DIRECTIONS * TOTAL_ZETA_NUMBERS) as u32 
            },
            mip_level_count:     1,
            sample_count:        1,
            dimension:           TextureDimension::D2,
            format:              TextureFormat::R32Float,
            usage:               TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            view_formats:        &[],
        },
        data: Some(vec![0u8; (GRID_X_SIZE * GRID_Z_SIZE * DIRECTIONS * TOTAL_ZETA_NUMBERS * 4) as usize]),
        ..default()
    };
    let amplitude_image_b = Image {
        texture_descriptor: TextureDescriptor {
            label:               Some("amplitude_texture"),
            size:                Extent3d {
                width: GRID_X_SIZE as u32,
                height: GRID_Z_SIZE as u32,
                depth_or_array_layers: (DIRECTIONS * TOTAL_ZETA_NUMBERS) as u32
            },
            mip_level_count:     1,
            sample_count:        1,
            dimension:           TextureDimension::D2,
            format:              TextureFormat::R32Float,
            usage:               TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            view_formats:        &[],
        },
        data: Some(vec![0u8; (GRID_X_SIZE * GRID_Z_SIZE * DIRECTIONS * TOTAL_ZETA_NUMBERS * 4) as usize]),
        ..default()
    };

    // Create placeholder profile buffer
    let profile_image = Image {
        texture_descriptor: TextureDescriptor {
            label:               Some("profile_buffer"),
            size:                Extent3d { width: PROFILE_BUFFER_LENGTH as u32, height: TOTAL_ZETA_NUMBERS as u32, depth_or_array_layers: 1 },
            mip_level_count:     1,
            sample_count:        1,
            dimension:           TextureDimension::D2,
            format:              TextureFormat::Rgba32Float,
            usage:               TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            view_formats:        &[],
        },
        data: Some(vec![0u8; (PROFILE_BUFFER_LENGTH * TOTAL_ZETA_NUMBERS * 16) as usize]),
        ..default()
    };

    commands.insert_resource(HeightFieldHandle{
        offsets_handle: images.add(offsets_image),
        normals_handle: images.add(normal_image)
    });

    commands.insert_resource(WaterSimulationHandles {
        amplitude_texture_a: images.add(amplitude_image_a),
        amplitude_texture_b: images.add(amplitude_image_b),
        profile_buffer: images.add(profile_image),
        flipped: false
    });

    // ── FFT Ocean Textures ──────────────────────────────────────────────────
    let fft_size = GRID_X_SIZE as u32; // 256

    let mk_rg32 = |label: String, data: Vec<u8>| Image {
        texture_descriptor: TextureDescriptor {
            label: Some(Box::leak(label.into_boxed_str())),
            size: Extent3d { width: fft_size, height: fft_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        },
        data: Some(data),
        ..default()
    };

    let mk_r32 = |label: String, data: Vec<u8>| Image {
        texture_descriptor: TextureDescriptor {
            label: Some(Box::leak(label.into_boxed_str())),
            size: Extent3d { width: fft_size, height: fft_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        },
        data: Some(data),
        ..default()
    };

    let mk_rgba32 = |label: &'static str| Image {
        texture_descriptor: TextureDescriptor {
            label: Some(label),
            size: Extent3d { width: fft_size, height: fft_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        },
        data: Some(vec![0u8; (fft_size * fft_size * 16) as usize]),
        ..default()
    };

    let n = (fft_size * fft_size) as usize;

    let mut cascade_vec: Vec<FftCascadeHandles> = Vec::with_capacity(FFT_CASCADE_COUNT);
    for i in 0..FFT_CASCADE_COUNT {
        let patch_len = fft_ocean_config.patch_lengths[i];
        let (spectrum_bytes, conj_bytes, freq_bytes) = generate_fft_initial_data(&fft_ocean_config, patch_len);
        cascade_vec.push(FftCascadeHandles {
            initial_spectrum:      images.add(mk_rg32(format!("fft_init_spec_{i}"),   spectrum_bytes)),
            initial_spectrum_conj: images.add(mk_rg32(format!("fft_init_conj_{i}"),   conj_bytes)),
            angular_frequency:     images.add(mk_r32 (format!("fft_angular_{i}"),     freq_bytes)),
            evolved_spectrum_a:    images.add(mk_rg32(format!("fft_evolved_a_{i}"),   vec![0u8; n * 8])),
            evolved_spectrum_b:    images.add(mk_rg32(format!("fft_evolved_b_{i}"),   vec![0u8; n * 8])),
            height_field:          images.add(mk_r32 (format!("fft_height_{i}"),      vec![0u8; n * 4])),
            slope_spectrum:        images.add(mk_rg32(format!("fft_slope_spec_{i}"),  vec![0u8; n * 8])),
            slope_field:           images.add(mk_rg32(format!("fft_slope_{i}"),       vec![0u8; n * 8])),
            displacement_spectrum: images.add(mk_rg32(format!("fft_disp_spec_{i}"),  vec![0u8; n * 8])),
            displacement_field:    images.add(mk_rg32(format!("fft_disp_{i}"),       vec![0u8; n * 8])),
        });
    }
    let cascades: [FftCascadeHandles; FFT_CASCADE_COUNT] = match cascade_vec.try_into() {
        Ok(arr) => arr,
        Err(_) => panic!("cascade count mismatch"),
    };

    commands.insert_resource(FftOceanHandles {
        cascades,
        normal_field: images.add(mk_rgba32("fft_normal_field")),
    });
}

// ─── Update Systems ──────────────────────────────────────────────────────────

fn tick_water_time(
    time: Res<Time>,
    mut settings: ResMut<WaterSettings>,
    mut fft_settings_array: ResMut<FftOceanSettingsArray>,
    fft_config: Res<FftOceanConfig>,
) {
    settings.elapsed_time = (time.elapsed_secs()*3.0) + 100.0;
    settings.delta_time = time.delta_secs()*3.0;
    let elapsed = time.elapsed_secs();
    for i in 0..FFT_CASCADE_COUNT {
        fft_settings_array.cascades[i].elapsed_time = elapsed;
        fft_settings_array.cascades[i].patch_length = fft_config.patch_lengths[i];
    }
}

fn debug_input_system(
    mut debug_settings: ResMut<WaterDebugSettings>,
    key: Res<ButtonInput<KeyCode>>,
) {
    // H key injects amplitude this frame
    debug_settings.inject = if key.pressed(KeyCode::KeyH) { 1.0 } else { 0.0 };

    // Arrow keys adjust scale and radius
    if key.pressed(KeyCode::ArrowUp) {
        debug_settings.scale += 0.5;
        println!("scale {}", debug_settings.scale);
    }
    if key.pressed(KeyCode::ArrowDown) {
        debug_settings.scale -= 0.5;
        println!("scale {}", debug_settings.scale);
    }
    if key.pressed(KeyCode::ArrowRight) {
        debug_settings.radius += 0.5;
        println!("radius {}", debug_settings.radius);
    }
    if key.pressed(KeyCode::ArrowLeft) {
        debug_settings.radius -= 0.5;
        println!("radius {}", debug_settings.radius);
    }

    // WASD moves spawn center
    if key.pressed(KeyCode::KeyA) {
        debug_settings.spawn_center_x -= 1.0;
        println!("spawn_center {:?},{:?}", debug_settings.spawn_center_x,debug_settings.spawn_center_z);
    }
    if key.pressed(KeyCode::KeyD) {
        debug_settings.spawn_center_x += 1.0;
        println!("spawn_center {:?},{:?}", debug_settings.spawn_center_x,debug_settings.spawn_center_z);
    }
    if key.pressed(KeyCode::KeyS) {
        debug_settings.spawn_center_z -= 1.0;
        println!("spawn_center {:?},{:?}", debug_settings.spawn_center_x,debug_settings.spawn_center_z);
    }
    if key.pressed(KeyCode::KeyW) {
        debug_settings.spawn_center_z += 1.0;
        println!("spawn_center {:?},{:?}", debug_settings.spawn_center_x,debug_settings.spawn_center_z);
    }
}


// pub const GRID_X_SIZE: usize = 64;
// pub const GRID_Z_SIZE: usize = 64;

// pub const GRID_X_SIZE: usize = 256;
// pub const GRID_Z_SIZE: usize = 256;

pub const GRID_X_SIZE: usize = 256;
pub const GRID_Z_SIZE: usize = 256;
pub const GRID_SCALE: f32 = 1.0; //how many meters each grid cell represents

pub const FFT_CASCADE_COUNT: usize = 3;
pub const FFT_PATCH_LENGTHS: [f32; FFT_CASCADE_COUNT] = [128.0, 64.0, 32.0];

// pub const DIRECTIONS: usize = 1;
pub const DIRECTIONS: usize = 32;
// pub const TOTAL_ZETA_NUMBERS: usize = 1;
pub const TOTAL_ZETA_NUMBERS: usize = 4;
pub const MIN_ZETA: f32 = -5.0;
pub const MAX_ZETA: f32 = 3.3;
// pub const MAX_ZETA: f32 = -3.0;
pub const WIND_SPEED: f32 = 3.0;
const GRAVITY: f32 = 9.81;

// pub const RENDER_X_SIZE: usize = 256;
// pub const RENDER_Z_SIZE: usize = 256;
// pub const RENDER_X_SIZE: usize = 2048;
// pub const RENDER_Z_SIZE: usize = 2048;
pub const RENDER_X_SIZE: usize = 64;
pub const RENDER_Z_SIZE: usize = 64;


pub const PROFILE_BUFFER_LENGTH: usize = 512;
// pub const PROFILE_BUFFER_LENGTH: usize = 4096;

//to be put in main world
#[derive(Resource, ExtractResource, Clone, ShaderType)]
pub struct WaterSettings {
    pub height_scale: f32,
    pub time_scale: f32,
    pub delta_time: f32,
    pub elapsed_time: f32,
}

//to be put in main world
#[derive(Resource, ExtractResource, Clone)]
pub struct HeightFieldHandle {
    pub offsets_handle: Handle<Image>,
    pub normals_handle: Handle<Image>
}

/// Debug settings for amplitude injection (extracted to render world)
#[derive(Resource, ExtractResource, Clone, ShaderType)]
pub struct WaterDebugSettings {
    pub spawn_center_x: f32,
    pub spawn_center_z: f32,
    /// Radius of injection circle in grid units
    pub radius: f32,
    /// Amplitude scale
    pub scale: f32,
    /// Whether injection is active this frame (1.0 = active, 0.0 = inactive)
    pub inject: f32,
    /// Padding for alignment
    pub _padding_a: f32,
    pub _padding_b: f32,
    pub _padding_c: f32
}

/// Voxel grid settings for boundary detection in advection shader
#[derive(Resource, ExtractResource, Clone, ShaderType)]
pub struct VoxelSettings {
    pub water_to_voxel: Mat4,  // Transform water grid index coords to voxel index coords
    pub voxel_size: f32,       // Size of each voxel
    pub grid_dims: UVec3,      // Voxel grid dimensions
}

#[derive(Resource, ExtractResource, Clone, ShaderType)]
pub struct WaterWaveGroupSpeeds{
    pub wave_group_speeds: [Vec4; TOTAL_ZETA_NUMBERS],
}

impl Default for VoxelSettings {
    fn default() -> Self {
        Self {
            water_to_voxel: Mat4::IDENTITY,
            voxel_size: 1.0,
            grid_dims: UVec3::ONE,
        }
    }
}

/// Handle to the voxel grid 3D texture and SDF texture
#[derive(Resource, ExtractResource, Clone)]
pub struct VoxelGridHandle {
    pub voxel: Option<Handle<Image>>,
    pub sdf: Option<Handle<Image>>,
}

/// Handles to water simulation textures for use in materials
#[derive(Resource, ExtractResource, Clone)]
pub struct WaterSimulationHandles {
    pub amplitude_texture_a: Handle<Image>,
    pub amplitude_texture_b: Handle<Image>,
    pub profile_buffer: Handle<Image>,
    pub flipped: bool,
}

/// FFT ocean GPU settings (extracted to render world as a uniform buffer)
#[derive(Resource, ExtractResource, Clone, ShaderType)]
pub struct FftOceanSettings {
    pub elapsed_time: f32,
    pub patch_length: f32,
    pub _padding_b: f32,
    pub _padding_c: f32,
}

// ─── FFT Ocean Precomputation ────────────────────────────────────────────────

#[derive(Resource)]
pub struct FftOceanConfig {
    pub grid_size: u32,
    pub patch_lengths: [f32; FFT_CASCADE_COUNT],
    pub wind_speed: f32,
    pub wind_dir: Vec2,
    pub amplitude: f32,
    pub gravity: f32,
    pub fetch: f32,
    pub gamma: f32,
}

/// Per-cascade FFT ocean texture handles
#[derive(Clone)]
pub struct FftCascadeHandles {
    pub initial_spectrum:       Handle<Image>,
    pub initial_spectrum_conj:  Handle<Image>,
    pub angular_frequency:      Handle<Image>,
    pub evolved_spectrum_a:     Handle<Image>,
    pub evolved_spectrum_b:     Handle<Image>,
    pub height_field:           Handle<Image>,
    pub slope_spectrum:         Handle<Image>,
    pub slope_field:            Handle<Image>,
    pub displacement_spectrum:  Handle<Image>,
    pub displacement_field:     Handle<Image>,
}

/// Handles to FFT ocean textures (all cascades)
#[derive(Resource, ExtractResource, Clone)]
pub struct FftOceanHandles {
    pub cascades: [FftCascadeHandles; FFT_CASCADE_COUNT],
    pub normal_field: Handle<Image>,
}

/// Per-cascade FFT settings (one FftOceanSettings per cascade, extracted to render world)
#[derive(Resource, ExtractResource, Clone)]
pub struct FftOceanSettingsArray {
    pub cascades: [FftOceanSettings; FFT_CASCADE_COUNT],
}

fn setup_voxel_fallback(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // let handle = crate::voxelize::null_texture(&mut images, TextureDimension::D3, TextureFormat::R8Uint);
    // commands.insert_resource(VoxelGridHandle(handle));
}

//NOTES ABOUT COORDINATE INDEXES
//NO NOTES
fn sync_voxel_grid(
    build_data: Res<BuildData>,
    mut voxel_settings: ResMut<VoxelSettings>,
    mut voxel_handle: ResMut<VoxelGridHandle>,
    mut images: ResMut<Assets<Image>>,
) {
    if let Some(ref grid) = build_data.voxel_grid {
        voxel_settings.voxel_size = grid.voxel_size;
        voxel_settings.grid_dims = UVec3::new(grid.dimensions.0, grid.dimensions.1, grid.dimensions.2);
        // voxel_settings.water_to_voxel = Mat4::from_translation((vec3(-(GRID_X_SIZE as f32),0.0,-(GRID_Z_SIZE as f32)/2.0)) + (<Vec3A as Into<Vec3>>::into(grid.offset))).mul_mat4(&Mat4::from_scale(Vec3::splat(grid.voxel_size))).inverse();
        // voxel_settings.water_to_voxel = Mat4::from_translation((-grid.offset).into()).mul_mat4(&Mat4::from_scale(Vec3::splat(1.0/grid.voxel_size))).mul_mat4(&Mat4::from_translation(vec3(-(GRID_X_SIZE as f32)/2.0,0.0,-(GRID_Z_SIZE as f32)/2.0)));
        // let grid_to_world = Mat4::from_translation(vec3(-(GRID_X_SIZE as f32)/2.0,0.0,-(GRID_Z_SIZE as f32)/2.0)).mul_mat4(&Mat4::from_scale(vec3(GRID_X_SIZE as f32/1000.0, 1.0, GRID_Z_SIZE as f32/1000.0)));
        // let grid_to_world = Mat4::from_translation(vec3(-(GRID_X_SIZE as f32)/2.0,0.0,-(GRID_Z_SIZE as f32)/2.0)).mul_mat4(&Mat4::from_scale(vec3(1000.0/GRID_X_SIZE as f32, 1.0, 1000.0/GRID_Z_SIZE as f32)));
        let grid_to_world = Mat4::from_scale(Vec3::splat(GRID_SCALE)).mul_mat4(&Mat4::from_translation(vec3(-(GRID_X_SIZE as f32)/2.0,0.0,-(GRID_Z_SIZE as f32)/2.0)));
        // let grid_to_world = Mat4::from_scale(vec3(GRID_X_SIZE as f32/1000.0, 1.0, GRID_Z_SIZE as f32/1000.0)).mul_mat4(&Mat4::from_translation(vec3(-(GRID_X_SIZE as f32)/2.0,0.0,-(GRID_Z_SIZE as f32)/2.0)));
        // voxel_settings.water_to_voxel = grid_to_world;
        // voxel_settings.water_to_voxel = Mat4::from_scale(Vec3::splat(1.0/grid.voxel_size)).mul_mat4(&Mat4::from_translation((-grid.offset).into())).mul_mat4(&grid_to_world);
        // voxel_settings.water_to_voxel = grid_to_world.mul_mat4(&Mat4::from_translation((-grid.offset).into())).mul_mat4(&Mat4::from_scale(Vec3::splat(1.0/grid.voxel_size)));
        // voxel_settings.water_to_voxel = grid_to_world.mul_mat4(&Mat4::from_scale(Vec3::splat(1.0/grid.voxel_size))).mul_mat4(&Mat4::from_translation((-grid.offset).into()));
        // voxel_settings.water_to_voxel = grid_to_world.mul_mat4(&Mat4::from_scale(Vec3::splat(1.0/grid.voxel_size)));
        voxel_settings.water_to_voxel = Mat4::from_scale(Vec3::splat(1.0/grid.voxel_size)).mul_mat4(&Mat4::from_translation((-grid.offset).into())).mul_mat4(&grid_to_world);
        // voxel_settings.water_to_voxel = grid_to_world;
        // voxel_settings.water_to_voxel = Mat4::IDENTITY;
        if voxel_handle.voxel.is_none() {
            voxel_handle.voxel = Some(crate::voxelize::create_voxel_texture(grid, &mut images));
            voxel_handle.sdf = Some(crate::voxelize::create_sdf_texture(grid, &mut images));
        }
    }
}