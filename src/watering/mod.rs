mod compute_shader;
mod cpu_debug_impl;
pub(crate) mod water_lod;
pub(crate) mod shader_utils;

use std::collections::HashMap;
use bevy::app::{App, Plugin, Startup, Update};
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{Extent3d, ShaderType, TextureFormat};
use crate::BuildData;
use bevy::render::{Render, RenderApp, RenderSystems};
use crate::watering::cpu_debug_impl::CpuDebugWatering;
use crate::watering::shader_utils::{compute_group_speeds, make_storage_image};

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
            .insert_resource(GpuTextures::default())
            .insert_resource(WaterPingPong::default())
            .insert_resource(WaterWaveGroupSpeeds {
                wave_group_speeds: compute_group_speeds(MIN_ZETA, MAX_ZETA, WIND_SPEED, TOTAL_ZETA_NUMBERS).iter().map(|x| vec4(*x * GRID_SCALE, 0.0, 0.0, 0.0)).collect::<Vec<_>>().try_into().unwrap()
            })
            .insert_resource(VoxelSettings::default())
            .add_systems(Startup, (setup_textures))
            .add_systems(Update, debug_input_system)
            .add_systems(Update, (tick_water_time, sync_voxel_grid))
        // Advance simulation time each frame
        // .add_systems(Update, (tick_water_time, sync_ship_position))
        ;

        if(false) {
            app
                .add_plugins(ExtractResourcePlugin::<WaterSettings>::default())
                .add_plugins(ExtractResourcePlugin::<WaterDebugSettings>::default())
                .add_plugins(ExtractResourcePlugin::<VoxelSettings>::default())
                .add_plugins(ExtractResourcePlugin::<WaterWaveGroupSpeeds>::default())
                .add_plugins(ExtractResourcePlugin::<GpuTextures>::default())
                .add_plugins(ExtractResourcePlugin::<WaterPingPong>::default())
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
            render_app.add_systems(
                Core3d,
                compute_shader::render_pass.in_set(Core3dSystems::Prepass)
            );
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
    mut images: ResMut<Assets<Image>>,
){
    let render_size = Extent3d {
        width: RENDER_X_SIZE as u32,
        height: RENDER_Z_SIZE as u32,
        depth_or_array_layers: 1,
    };
    let amplitude_size = Extent3d {
        width: GRID_X_SIZE as u32,
        height: GRID_Z_SIZE as u32,
        depth_or_array_layers: (DIRECTIONS * TOTAL_ZETA_NUMBERS) as u32,
    };
    let profile_size = Extent3d {
        width: PROFILE_BUFFER_LENGTH as u32,
        height: TOTAL_ZETA_NUMBERS as u32,
        depth_or_array_layers: 1,
    };

    let mut textures = GpuTextures::default();
    textures.insert(
        GpuTexture::HeightOffsets,
        images.add(make_storage_image("height_field_mat", render_size, TextureFormat::Rgba32Float)),
    );
    textures.insert(
        GpuTexture::HeightNormals,
        images.add(make_storage_image("normal_field_mat", render_size, TextureFormat::Rgba32Float)),
    );
    textures.insert(
        GpuTexture::AmplitudeA,
        images.add(make_storage_image("amplitude_texture_a", amplitude_size, TextureFormat::R32Float)),
    );
    textures.insert(
        GpuTexture::AmplitudeB,
        images.add(make_storage_image("amplitude_texture_b", amplitude_size, TextureFormat::R32Float)),
    );
    textures.insert(
        GpuTexture::Profile,
        images.add(make_storage_image("profile_buffer", profile_size, TextureFormat::Rgba32Float)),
    );

    commands.insert_resource(textures);
}

// ─── Update Systems ──────────────────────────────────────────────────────────

fn tick_water_time(time: Res<Time>, mut settings: ResMut<WaterSettings>) {
    // settings.elapsed_time = time.elapsed_secs();
    settings.elapsed_time = (time.elapsed_secs()*3.0) + 100.0;
    // settings.elapsed_time = 0.0;
    settings.delta_time = time.delta_secs()*3.0;
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

// pub const GRID_X_SIZE: usize = 256;
// pub const GRID_Z_SIZE: usize = 256;
pub const GRID_X_SIZE: usize = 64;
pub const GRID_Z_SIZE: usize = 64;
pub const GRID_SCALE: f32 = 1.0; //how many meters each grid cell represents

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

/// Keys for every `Handle<Image>` the watering pipeline owns.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum GpuTexture {
    HeightOffsets,
    HeightNormals,
    AmplitudeA,
    AmplitudeB,
    Profile,
    Voxel,
    Sdf,
}

/// Single resource for every texture handle used by the watering pipeline.
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct GpuTextures {
    pub map: HashMap<GpuTexture, Handle<Image>>,
}

impl GpuTextures {
    pub fn get(&self, key: GpuTexture) -> Option<&Handle<Image>> {
        self.map.get(&key)
    }
    pub fn insert(&mut self, key: GpuTexture, handle: Handle<Image>) {
        self.map.insert(key, handle);
    }
}

/// Ping-pong state for the two amplitude textures. Flipped each frame.
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct WaterPingPong {
    pub flipped: bool,
}

impl WaterPingPong {
    /// Returns (read_key, write_key) for the amplitude ping-pong this frame.
    pub fn amplitude_keys(&self) -> (GpuTexture, GpuTexture) {
        if self.flipped {
            (GpuTexture::AmplitudeB, GpuTexture::AmplitudeA)
        } else {
            (GpuTexture::AmplitudeA, GpuTexture::AmplitudeB)
        }
    }
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

//NOTES ABOUT COORDINATE INDEXES
//NO NOTES
fn sync_voxel_grid(
    build_data: Res<BuildData>,
    mut voxel_settings: ResMut<VoxelSettings>,
    mut textures: ResMut<GpuTextures>,
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
        if textures.get(GpuTexture::Voxel).is_none() {
            textures.insert(GpuTexture::Voxel, crate::voxelize::create_voxel_texture(grid, &mut images));
            textures.insert(GpuTexture::Sdf, crate::voxelize::create_sdf_texture(grid, &mut images));
        }
    }
}