mod node;
pub(crate) mod ocean_mesh;
mod pipeline;
mod resources;
mod buoyancy;

pub use ocean_mesh::{WaterMesh, spawn_water_mesh};
pub use resources::ObstacleMask;

use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        Render, RenderApp,
    },
};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::render::render_graph::RenderGraphExt;
use bevy::render::render_resource::ShaderType;
use bevy::render::RenderSystems;
use crate::water::CustomMaterial;
use crate::water_stuff;
// ─── Simulation Constants ────────────────────────────────────────────────────

/// Number of discrete wave propagation directions.
/// The paper uses 16; 8 is a good balance of quality vs. cost.
/// Must match NUM_DIRECTIONS in all WGSL shaders.
pub const NUM_DIRECTIONS: u32 = 8;

/// Amplitude grid resolution (cells per axis). Each cell covers
/// DOMAIN_SIZE / GRID_SIZE meters. 512 → ~2m/cell for a 1km domain.
pub const GRID_SIZE: u32 = 512;

/// Physical extent of the simulation domain in meters.
pub const DOMAIN_SIZE: f32 = 1024.0;

/// Number of samples in the 1-D profile buffer Ψ(p).
pub const PROFILE_SIZE: u32 = 1024;

/// Deep-water dominant wavenumber k₀ (rad/m).
/// Corresponds to a wavelength of 2π/k₀ ≈ 63 m.
/// Tune this to shift the dominant wave size.
pub const K0: f32 = 0.1;

// ─── Settings Resource ───────────────────────────────────────────────────────

/// Simulation parameters. Modify these at runtime to change wave behaviour.
/// Automatically extracted to the render world each frame.
#[derive(Resource, ExtractResource, Clone, ShaderType)]
pub struct WaterSettings {
    /// Physical size of the simulation domain (m).
    pub domain_size:    f32,
    /// Current simulation time (s). Updated automatically.
    pub time:           f32,
    /// Last frame delta time (s), capped at 1/30 s.
    pub dt:             f32,
    /// Dominant wavenumber k₀ (rad/m).
    pub k0:             f32,
    /// Gravity constant (m/s²).
    pub gravity:        f32,
    /// Wind direction (radians from +X).
    pub wind_direction: f32,
    /// Wind speed (m/s) – controls Phillips spectrum energy.
    pub wind_speed:     f32,
    /// Strength of along-travel (spatial) diffusion [0..1].
    pub diffusion_spatial:  f32,
    /// Strength of angular diffusion [0..1].
    pub diffusion_angular:  f32,
    /// Ship world-space position (XZ plane).
    pub ship_pos_x:       f32,
    /// Ship world-space position (XZ plane).
    pub ship_pos_z:       f32,
    /// How strongly the ship disturbs the amplitude field.
    pub ship_source_strength: f32,
    /// Grid size (for shader indexing).
    pub grid_size:      u32,
    /// Number of directions (for shader indexing).
    pub num_directions: u32,
    /// Profile buffer size.
    pub profile_size:   u32,
}

impl Default for WaterSettings {
    fn default() -> Self {
        Self {
            domain_size:          DOMAIN_SIZE,
            time:                 0.0,
            dt:                   1.0 / 60.0,
            k0:                   K0,
            gravity:              9.81,
            wind_direction:       0.0,
            wind_speed:           12.0,
            diffusion_spatial:    0.05,
            diffusion_angular:    0.02,
            ship_pos_x:           0.0,
            ship_pos_z:           0.0,
            ship_source_strength: 0.3,
            grid_size:            GRID_SIZE,
            num_directions:       NUM_DIRECTIONS,
            profile_size:         PROFILE_SIZE,
        }
    }
}

// ─── Plugin ──────────────────────────────────────────────────────────────────

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(ExtractResourcePlugin::<WaterSettings>::default())
            // Advance simulation time each frame
            // .add_systems(Update, (tick_water_time, sync_ship_position))
        ;

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<ObstacleMask>()
            .init_resource::<WaterSettings>()
            .add_systems(Update, (tick_water_time))
            .add_systems(Startup, (temp_setup))
            .add_systems(
                Render,
                (
                    resources::prepare_gpu_resources
                        .in_set(RenderSystems::PrepareResources),
                    resources::upload_obstacle_mask
                        .in_set(RenderSystems::PrepareResources)
                        .after(resources::prepare_gpu_resources),
                    pipeline::queue_bind_groups
                        .in_set(RenderSystems::Queue),
                ),
            );

        // use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
        // use bevy::render::render_graph::RenderGraphApp;
        render_app
            .add_render_graph_node::<node::WaterSimulationNode>(
                Core3d,
                node::WaterNodeLabel,
            )
            // Run our compute passes before the main 3D pass
            .add_render_graph_edge(Core3d, node::WaterNodeLabel, Node3d::EarlyPrepass);
    }

    fn finish(&self, app: &mut App) {
        // Pipeline creation needs the RenderDevice, which is only available
        // after the renderer is initialised (in `finish`).
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<pipeline::WaterPipelines>();
    }
}

fn temp_setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut water_material: ResMut<Assets<water_stuff::ocean_mesh::WaterMaterial>>,
){
    spawn_water_mesh(&mut commands, &mut meshes, &mut water_material, &mut images);
}

// ─── Update Systems ──────────────────────────────────────────────────────────

fn tick_water_time(time: Res<Time>, mut settings: ResMut<WaterSettings>) {
    settings.time += time.delta_secs();
    settings.dt = time.delta_secs().clamp(0.0, 1.0 / 30.0);
}

/// Marker component – attach this to your ship entity so the plugin
/// can track its world-space position automatically.
#[derive(Component)]
pub struct ShipMarker;

fn sync_ship_position(
    ships: Query<&Transform, With<ShipMarker>>,
    mut settings: ResMut<WaterSettings>,
) {
    if let Ok(t) = ships.single() {
        settings.ship_pos_x = t.translation.x;
        settings.ship_pos_z = t.translation.z;
    }
}
