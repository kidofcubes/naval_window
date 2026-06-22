//! CPU implementations of water simulation compute shaders for debugging purposes.
//! These mirror the GPU shader logic for validation and debugging.

use std::f32::consts::*;
use bevy::app::hotpatch::call;
use bevy::light::NotShadowCaster;
use bevy::prelude::*;
use bevy::render::render_resource::Face;
use super::{
    GRID_X_SIZE, GRID_Z_SIZE, RENDER_X_SIZE, RENDER_Z_SIZE,
    DIRECTIONS, TOTAL_ZETA_NUMBERS, PROFILE_BUFFER_LENGTH,
    MIN_ZETA, MAX_ZETA, GRAVITY,
    WaterSettings, WaterDebugSettings,
};
use crate::utils::{build_grid_mesh, sample_bilinear_vec4};

/// Directional offsets for phase coherence (matches water_common.wgsl)
// const DIRECTIONAL_OFFSETS: [f32; 64] = [
//     0.0,
//     0.09817477042468103,
//     0.19634954084936207,
//     0.2945243112740431,
//     0.39269908169872414,
//     0.4908738521234052,
//     0.5890486225480862,
//     0.6872233929727672,
//     0.7853981633974483,
//     0.8835729338221293,
//     0.9817477042468103,
//     1.0799224746714913,
//     1.1780972450961724,
//     1.2762720155208536,
//     1.3744467859455345,
//     1.4726215563702154,
//     1.5707963267948966,
//     1.6689710972195777,
//     1.7671458676442586,
//     1.8653206380689396,
//     1.9634954084936207,
//     2.061670178918302,
//     2.1598449493429825,
//     2.2580197197676637,
//     2.356194490192345,
//     2.454369260617026,
//     2.552544031041707,
//     2.650718801466388,
//     2.748893571891069,
//     2.84706834231575,
//     2.945243112740431,
//     3.043417883165112,
//     3.141592653589793,
//     3.2397674240144743,
//     3.3379421944391554,
//     3.436116964863836,
//     3.5342917352885173,
//     3.6324665057131984,
//     3.730641276137879,
//     3.8288160465625602,
//     3.9269908169872414,
//     4.025165587411922,
//     4.123340357836604,
//     4.221515128261284,
//     4.319689898685965,
//     4.417864669110647,
//     4.516039439535327,
//     4.614214209960009,
//     4.71238898038469,
//     4.81056375080937,
//     4.908738521234052,
//     5.006913291658733,
//     5.105088062083414,
//     5.203262832508095,
//     5.301437602932776,
//     5.399612373357457,
//     5.497787143782138,
//     5.595961914206819,
//     5.6941366846315,
//     5.792311455056181,
//     5.890486225480862,
//     5.988660995905543,
//     6.086835766330224,
//     6.1850105367549055
// ];
const DIRECTIONAL_OFFSETS: [f32; 64] = [0.0; 64];

// because of some stuff, higher values of this make smaller profile
const INTEGRATION_STEPS: usize = 64;
const PERIODICITY: f32 = 2.0;

/// Convert direction index to radians
#[inline]
fn index_to_radians(i: usize) -> f32 {
    (i as f32 + 0.0) * (TAU / DIRECTIONS as f32)
}

/// Compute layer index for amplitude texture array
#[inline]
fn layer_index(theta_index: usize, zeta_index: usize) -> usize {
    zeta_index * DIRECTIONS + theta_index
}

/// Modulo that handles negative numbers correctly
#[inline]
fn modulo(x: f32, n: f32) -> f32 {
    x - ((x / n).floor() * n)
}

// SYSTEMS
#[derive(Resource)]
pub struct CpuDebugWateringResources {
    pub amplitude_buffer: AmplitudeBuffer, //amplitudes for each zeta and direction and position
    pub profile_buffer: ProfileBuffer, //for each zeta index and then profile length
    pub height_buffer: HeightBuffer,
    pub normal_buffer: NormalBuffer,
}

#[derive(Component)]
#[require(Transform)]
struct DebugWaterMesh;

const DEBUG_MESH_OFFSET: Vec3 = Vec3::new(0.0, 1.0, 0.0);
/// How many mesh vertices per height-buffer cell along each axis.
/// 1 = same resolution as the buffer, 4 = 4× denser per axis (16× vertex count).
const DEBUG_MESH_UPSAMPLE: usize = 4;
const DEBUG_MESH_COLS: usize = RENDER_X_SIZE * DEBUG_MESH_UPSAMPLE;
const DEBUG_MESH_ROWS: usize = RENDER_Z_SIZE * DEBUG_MESH_UPSAMPLE;
const DEBUG_MESH_CELL_X: f32 = 1.0 / DEBUG_MESH_COLS as f32;
const DEBUG_MESH_CELL_Z: f32 = 1.0 / DEBUG_MESH_ROWS as f32;
const DEBUG_MESH_DISP_SCALE: f32 = 1.0 / RENDER_X_SIZE as f32;

pub struct CpuDebugWatering;
impl Plugin for CpuDebugWatering {
    fn build(&self, app: &mut App) {
        app.insert_resource(CpuDebugWateringResources {
            amplitude_buffer: new_amplitude_buffer(),
            profile_buffer: new_profile_buffer(),
            height_buffer: new_height_buffer(),
            normal_buffer: new_normal_buffer(),
        });
        app.add_systems(Startup, init_debug_water_mesh);
        app.add_systems(Update, update);
    }
}

fn init_debug_water_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = build_grid_mesh(
        DEBUG_MESH_COLS,
        DEBUG_MESH_ROWS,
        DEBUG_MESH_CELL_X,
        DEBUG_MESH_CELL_Z,
        Vec2::ZERO,
    );
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.5, 0.5),
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            cull_mode: Some(Face::Back),
            perceptual_roughness: 1.0,
            reflectance: 0.0,
            metallic: 0.0,
            ..default()
        })),
        NotShadowCaster,
        Transform::from_translation(DEBUG_MESH_OFFSET),
        DebugWaterMesh,
    ));
}

fn get_offset_for_position(render_pos: Vec2, zeta_index: usize, theta_index: usize, profile: &ProfileBuffer) -> Vec2 {
    let grid_pos = vec2(
        render_pos.x * (RENDER_X_SIZE as f32 / GRID_X_SIZE as f32),
        render_pos.y * (RENDER_Z_SIZE as f32 / GRID_Z_SIZE as f32)
    );
    let theta = index_to_radians(theta_index);
    let dir = vec3(f32::cos(theta), 0.0, f32::sin(theta));

    let zeta_full_range = MAX_ZETA - MIN_ZETA;
    let wind_speed = 3.0; // Hardcoded in shader

    let min_zeta = MIN_ZETA + (zeta_full_range * ((zeta_index as f32 + 0.0) / TOTAL_ZETA_NUMBERS as f32));
    let max_zeta = MIN_ZETA + (zeta_full_range * ((zeta_index as f32 + 1.0) / TOTAL_ZETA_NUMBERS as f32));
    let mid_zeta = MIN_ZETA + (zeta_full_range * ((zeta_index as f32 + 0.5) / TOTAL_ZETA_NUMBERS as f32));

    let period = PERIODICITY * 2.0f32.powf(mid_zeta);



    // Position offset along wave direction
    let position_offset = grid_pos.dot(vec2(dir.x,dir.z));

    // Profile buffer index (wrapping)
    let profile_coord = position_offset + DIRECTIONAL_OFFSETS[theta_index];
    let index = modulo(
        PROFILE_BUFFER_LENGTH as f32 * (profile_coord / period),
        PROFILE_BUFFER_LENGTH as f32,
    ) as usize;
    let index = index.min(PROFILE_BUFFER_LENGTH - 1);

    let mut profile_value = profile[zeta_index][index];

    let mut offset: Vec2 = Vec2::ZERO;
    let amplitude = vec2(1.0,0.0);
    let cos_val = profile_value;

    offset.y += (amplitude.x * cos_val.y) - (amplitude.y * -cos_val.x); //vertical

    offset.x += (amplitude.x * cos_val.x) - (amplitude.y *  cos_val.y); //horizontal

    return offset;
}

fn update(
    mut cpu_debug_watering_resources: ResMut<CpuDebugWateringResources>,
    mut water_debug_settings: ResMut<WaterDebugSettings>,
    mut water_settings: ResMut<WaterSettings>,
    mut gizmos: Gizmos,
    time: Res<Time>,
    key: Res<ButtonInput<KeyCode>>,
    debug_mesh_query: Query<&Mesh3d, With<DebugWaterMesh>>,
    mut meshes: ResMut<Assets<Mesh>>,
){

    if(key.pressed(KeyCode::KeyC)) {

    }
    // if(key.pressed(KeyCode::KeyH)) {
        water_debug_settings.inject = 1.0;
        let cpu_debug_watering_resources = cpu_debug_watering_resources.as_mut();

        let delta_time = time.delta_secs() * 3.0;
        // water_settings.elapsed_time = 100.0;
        water_settings.elapsed_time = (water_debug_settings.spawn_center_z * 0.01) + 100.0;
        // water_settings.elapsed_time = (time.elapsed_secs() * 3.0) + 100.0;
        // water_settings.elapsed_time = 0.0;

        call(|| build_profile(&mut cpu_debug_watering_resources.profile_buffer, &water_settings));
        call(|| inject_amplitude(&mut cpu_debug_watering_resources.amplitude_buffer, &cpu_debug_watering_resources.profile_buffer, &water_settings, &water_debug_settings));
        call(|| reconstruct_height(&cpu_debug_watering_resources.amplitude_buffer, &cpu_debug_watering_resources.profile_buffer, &mut cpu_debug_watering_resources.height_buffer, &mut cpu_debug_watering_resources.normal_buffer, &water_settings, Some(&mut gizmos)));
        // inject_amplitude(&mut cpu_debug_watering_resources.amplitude_buffer, &water_settings, &water_debug_settings);
        // build_profile(&mut cpu_debug_watering_resources.profile_buffer, &water_settings);
        // reconstruct_height(&cpu_debug_watering_resources.amplitude_buffer, &cpu_debug_watering_resources.profile_buffer, &mut cpu_debug_watering_resources.height_buffer, &mut cpu_debug_watering_resources.normal_buffer, &water_settings);
    // }
    if(key.just_pressed(KeyCode::KeyC)){
        for i in 0..(TOTAL_ZETA_NUMBERS*DIRECTIONS){
            for x in 0..GRID_X_SIZE {
                for z in 0..GRID_Z_SIZE {
                    cpu_debug_watering_resources.amplitude_buffer[i][x][z] = vec2(0.0, 0.0);
                }
            }
        }
    }

    for zeta_index in 0..TOTAL_ZETA_NUMBERS {
        let offset = vec3(-2.0 + (zeta_index as f32 * -1.0), 1.0, 0.0);
        for i in 0..PROFILE_BUFFER_LENGTH {
            let num = cpu_debug_watering_resources.profile_buffer[zeta_index][i];
            let pos = offset + vec3(0.0, 0.0, (i as f32)/100.0);
            gizmos.line(
                pos,
                pos + vec3(0.0, num.x, 0.0),
                Color::srgba(1.0, 0.0, (i as f32 / PROFILE_BUFFER_LENGTH as f32), 1.0),
            );
            gizmos.line(
                pos,
                // pos + vec3(0.1, 0.0, 0.0),
                pos + vec3(num.y, 0.0, 0.0),
                // pos + vec3(0.1, cpu_debug_watering_resources.profile_buffer[zeta_index][(i+(((6*PROFILE_BUFFER_LENGTH)/4)))%PROFILE_BUFFER_LENGTH].x, 0.0),
                Color::srgba(1.0, 0.0, (i as f32 / PROFILE_BUFFER_LENGTH as f32), 1.0),
            );
        }

        gizmos.text(Isometry3d {
            translation: (offset + vec3(0.0, 0.0, -0.1)).into(),
            ..default()
        }, &format!("{:?}",zeta_index), 0.1, Vec2::new(0.5,0.5), Color::srgba_u8(0,255,0,255));
        for i in 0..zeta_index {
            let pos = offset + vec3(0.0, 0.0, -0.05+(-0.05*i as f32));
            // gizmos.line(
            //     pos,
            //     // pos + vec3(0.1, 0.0, 0.0),
            //     pos + vec3(-0.1, 0.0, 0.0),
            //     // pos + vec3(0.1, cpu_debug_watering_resources.profile_buffer[zeta_index][(i+(((6*PROFILE_BUFFER_LENGTH)/4)))%PROFILE_BUFFER_LENGTH].x, 0.0),
            //     Color::srgba(1.0, 0.0, 1.0, 1.0),
            // )
            // gizmos.
        }
    }

    let offset = vec3(-0.0, 1.0, 0.0);
    gizmos.text(Isometry3d {
        translation: (offset + vec3(0.0, 1.0, 0.0)).into(),
        ..default()
    }, &format!("x0 z0"), 0.1, Vec2::new(0.5,0.5), Color::srgba_u8(0,255,0,255));

    gizmos.text(Isometry3d {
        translation: (offset + vec3(1.0, 1.0, 0.0)).into(),
        ..default()
    }, &format!("x1 z0"), 0.1, Vec2::new(0.5,0.5), Color::srgba_u8(0,255,0,255));

    gizmos.text(Isometry3d {
        translation: (offset + vec3(0.0, 1.0, 1.0)).into(),
        ..default()
    }, &format!("x0 z1"), 0.1, Vec2::new(0.5,0.5), Color::srgba_u8(0,255,0,255));

    gizmos.text(Isometry3d {
        translation: (offset + vec3(1.0, 1.0, 1.0)).into(),
        ..default()
    }, &format!("x1 z1"), 0.1, Vec2::new(0.5,0.5), Color::srgba_u8(0,255,0,255));



    for x in 0..RENDER_X_SIZE {
        // for z in 0..RENDER_Z_SIZE {
        for z in 0..16 {
            let pos = vec3(x as f32/RENDER_X_SIZE as f32, 0.0, z as f32/RENDER_Z_SIZE as f32) + offset;
            let offset = cpu_debug_watering_resources.height_buffer[z][x];
            let normal = cpu_debug_watering_resources.normal_buffer[z][x];
            gizmos.line(
                pos,
                pos + (offset.xyz() /(RENDER_X_SIZE as f32 * 1.0)),
                Color::srgba(1.0, x as f32/RENDER_X_SIZE as f32, z as f32/RENDER_Z_SIZE as f32, 1.0),
            );

            gizmos.line(
                pos,
                pos + (normal.xyz()*0.01),
                Color::srgba(x as f32/RENDER_X_SIZE as f32, 1.0, z as f32/RENDER_Z_SIZE as f32, 1.0),
            );
            if(x==0&&z==0){
                gizmos.text(Isometry3d {
                    translation: (pos + vec3(0.0, 0.5, 0.0)).into(),
                    ..default()
                }, &format!("x{} z{} height {} normal {}", x, z, offset, normal), 0.1, Vec2::new(0.5,0.5), Color::srgba_u8(0,255,0,255));
            }
        }
    }

    if let Ok(mesh_handle) = debug_mesh_query.single() {
        if let Some(mut mesh) = meshes.get_mut(mesh_handle) {
            let inv_upsample = 1.0 / DEBUG_MESH_UPSAMPLE as f32;
            if let Some(positions) = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION) {
                if let bevy::mesh::VertexAttributeValues::Float32x3(pos) = positions {
                    for mesh_row in 0..DEBUG_MESH_ROWS {
                        for mesh_col in 0..DEBUG_MESH_COLS {
                            let frac_col = mesh_col as f32 * inv_upsample;
                            let frac_row = mesh_row as f32 * inv_upsample;
                            let surface = sample_bilinear_vec4(
                                &cpu_debug_watering_resources.height_buffer,
                                frac_col,
                                frac_row,
                            );
                            let base_x = mesh_col as f32 * DEBUG_MESH_CELL_X;
                            let base_z = mesh_row as f32 * DEBUG_MESH_CELL_Z;
                            pos[mesh_row * DEBUG_MESH_COLS + mesh_col] = [
                                base_x + surface.x * DEBUG_MESH_DISP_SCALE,
                                surface.y * DEBUG_MESH_DISP_SCALE,
                                base_z + surface.z * DEBUG_MESH_DISP_SCALE,
                            ];
                        }
                    }
                }
            }
            if let Some(normal_attr) = mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL) {
                if let bevy::mesh::VertexAttributeValues::Float32x3(mesh_normals) = normal_attr {
                    for mesh_row in 0..DEBUG_MESH_ROWS {
                        for mesh_col in 0..DEBUG_MESH_COLS {
                            let frac_col = mesh_col as f32 * inv_upsample;
                            let frac_row = mesh_row as f32 * inv_upsample;
                            let n = sample_bilinear_vec4(
                                &cpu_debug_watering_resources.normal_buffer,
                                frac_col,
                                frac_row,
                            );
                            let n3 = bevy::math::vec3(n.x, n.y, n.z).normalize_or_zero();
                            mesh_normals[mesh_row * DEBUG_MESH_COLS + mesh_col] = [n3.x, n3.y, n3.z];
                        }
                    }
                }
            }
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// WATER INJECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Amplitude storage: [layer][z][x] where layer = zeta_index * DIRECTIONS + theta_index
/// Each amplitude is (real, imag) = (cos, sin) component
pub type AmplitudeBuffer = Vec<Vec<Vec<Vec2>>>;

/// Create a new amplitude buffer initialized to zero
pub fn new_amplitude_buffer() -> AmplitudeBuffer {
    let num_layers = TOTAL_ZETA_NUMBERS * DIRECTIONS;
    vec![vec![vec![Vec2::ZERO; GRID_X_SIZE]; GRID_Z_SIZE]; num_layers]
}

/// CPU implementation of inject_amplitude compute shader.
/// Injects phase-coherent waves in a circular region for debugging.
pub fn inject_amplitude(
    amplitude_out: &mut AmplitudeBuffer,
    profile_in: &ProfileBuffer,
    water_settings: &WaterSettings,
    debug_settings: &WaterDebugSettings,
) {
    // Skip if injection is not active
    if debug_settings.inject < 0.5 {
        return;
    }

    let zeta_full_range = MAX_ZETA - MIN_ZETA;

    for z in 0..GRID_Z_SIZE {
        for x in 0..GRID_X_SIZE {
            let pos_x = x as f32;
            let pos_z = z as f32;
            let dist = ((pos_x - debug_settings.spawn_center_x).powi(2)
                + (pos_z - debug_settings.spawn_center_z).powi(2))
            .sqrt();

            // Skip if outside injection radius
            // if dist > debug_settings.radius {
            //     continue;
            // }

            // for theta_index in 0..DIRECTIONS {
            for theta_index in 0..1{
                // let theta_index = 0;
                let theta = index_to_radians(theta_index);
                let dir = vec3(f32::cos(theta), 0.0, f32::sin(theta));

                for zeta_index in 0..TOTAL_ZETA_NUMBERS {
                    let min_zeta = MIN_ZETA + (zeta_full_range * (((zeta_index as f32) - 0.0) / (TOTAL_ZETA_NUMBERS as f32)));
                    let max_zeta = MIN_ZETA + (zeta_full_range * (((zeta_index as f32) + 1.0) / (TOTAL_ZETA_NUMBERS as f32)));
                    let mid_zeta = MIN_ZETA + (zeta_full_range * ((zeta_index as f32 + 0.5) / TOTAL_ZETA_NUMBERS as f32));
                    // let period = PERIODICITY * 2.0f32.powf(max_zeta);
                    let period = PERIODICITY * 2.0f32.powf(mid_zeta);

                    let wave_length0 = 2.0f32.powf(min_zeta);
                    let wave_number0 = TAU / wave_length0;

                    let wave_length1 = 2.0f32.powf(mid_zeta);
                    let wave_number1 = TAU / wave_length1;

                    let wave_length2 = 2.0f32.powf(max_zeta);
                    let wave_number2 = TAU / wave_length2;

                    // Position offset along wave direction
                    let position_offset = vec2(pos_x,pos_z).dot(vec2(dir.x,dir.z));

                    // profile coord from height reconstruction
                    let profile_coord = (position_offset + DIRECTIONAL_OFFSETS[theta_index]);
                    // index used by height reconstruction
                    let index = modulo(
                        PROFILE_BUFFER_LENGTH as f32 * (profile_coord / period),
                        PROFILE_BUFFER_LENGTH as f32,
                    ) as usize;

                    let spatial_phase =
                        (wave_number1 * profile_coord)
                        - (dispersion_relation(wave_number1) * water_settings.elapsed_time);
                    let mut phase_coordinate = spatial_phase;

                    // Create amplitude with correct phase
                    // Note: shader stores (scale, 0.0) - simplified version
                    let layer = layer_index(theta_index, zeta_index);
                    let mut best_index = index;

                    // for i in (PROFILE_BUFFER_LENGTH as i32)/-8..(PROFILE_BUFFER_LENGTH as i32)/8 {
                    //     let shifted_index = modulo(index+i as f32,PROFILE_BUFFER_LENGTH as f32) as usize;
                    //     let shifted_index = shifted_index.min(PROFILE_BUFFER_LENGTH - 1);
                    // }
                    if debug_settings.radius > 0.0 {
                        phase_coordinate = 0.0;
                        let render_pos = vec2(
                            (pos_x / GRID_X_SIZE as f32) * RENDER_X_SIZE as f32,
                            (pos_z / GRID_Z_SIZE as f32) * RENDER_Z_SIZE as f32,
                        );
                        let orig_offset = get_offset_for_position(render_pos, zeta_index, theta_index, profile_in);
                        let orig_angle = f32::atan2(orig_offset.y, orig_offset.x);
                        phase_coordinate = orig_angle - PI/2.0;
                    }

                    let amplitude = vec2(
                        f32::cos((-phase_coordinate)+(debug_settings.spawn_center_x*0.01)),
                        // f32::cos((-0.0)+(debug_settings.spawn_center_x*0.01)),
                        f32::sin((-phase_coordinate)+(debug_settings.spawn_center_x*0.01))
                        // f32::sin((-0.0)+(debug_settings.spawn_center_x*0.01))
                    );

                    amplitude_out[layer][z][x] = amplitude * debug_settings.scale;
                    // amplitude_out[layer][z][x] = vec2(debug_settings.scale, 0.0);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WATER PROFILE GENERATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Profile buffer: [zeta_index][profile_index] -> (horizontal_disp, vertical_height, d_horiz/dp, d_vert/dp)
pub type ProfileBuffer = Vec<Vec<Vec4>>;

/// Create a new profile buffer initialized to zero
pub fn new_profile_buffer() -> ProfileBuffer {
    vec![vec![Vec4::ZERO; PROFILE_BUFFER_LENGTH]; TOTAL_ZETA_NUMBERS]
}

/// Pierson-Moskowitz spectrum function
fn spectrum(zeta: f32, wind_speed: f32) -> f32 {
    let a = 1.1f32.powf(1.5 * zeta);
    let b = (-1.8038897788076411 * 4.0f32.powf(zeta) / wind_speed.powi(4)).exp();
    0.139098 * (a * b).sqrt()
}

/// Dispersion relation (angular rotation): omega = sqrt(g * k)
#[inline]
fn dispersion_relation(k: f32) -> f32 {
    (GRAVITY * k).sqrt()
}

/// Cubic bump function for smooth blending
fn cubic_bump(x: f32) -> f32 {
    if x.abs() >= 1.0 {
        return 0.0;
    }
    x * x * ((2.0 * x.abs()) - 3.0) + 1.0
}

/// Gerstner wave contribution
/// Returns (horizontal_displacement, vertical_height, d_horiz/dp, d_vert/dp)
fn gerstner_wave(phase: f32, knum: f32) -> Vec4 {
    let s = phase.sin();
    let c = phase.cos();
    Vec4::new(
        -s,          // horizontal displacement
        c,           // vertical height
        -knum * c,   // d(horizontal)/dp
        -knum * s,   // d(vertical)/dp
    )
}

/// CPU implementation of build_profile compute shader.
/// Pre-computes the Gerstner wave profile for each zeta band.
pub fn build_profile(profile_out: &mut ProfileBuffer, settings: &WaterSettings) {
    let zeta_full_range = MAX_ZETA - MIN_ZETA;
    let wind_speed = 3.0; // Hardcoded in shader

    for zeta_index in 0..TOTAL_ZETA_NUMBERS {
        let min_zeta = MIN_ZETA + (zeta_full_range * ((zeta_index as f32 + 0.0) / TOTAL_ZETA_NUMBERS as f32));
        let max_zeta = MIN_ZETA + (zeta_full_range * ((zeta_index as f32 + 1.0) / TOTAL_ZETA_NUMBERS as f32));
        let mid_zeta = MIN_ZETA + (zeta_full_range * ((zeta_index as f32 + 0.5) / TOTAL_ZETA_NUMBERS as f32));

        for profile_index in 0..PROFILE_BUFFER_LENGTH {
            let mut result = Vec4::ZERO;

            let dx = (max_zeta - min_zeta) / INTEGRATION_STEPS as f32;

            for i in 0..INTEGRATION_STEPS {
                let zeta =
                    min_zeta + ((max_zeta - min_zeta) * ((i as f32 + 0.5) / INTEGRATION_STEPS as f32));
                // let zeta = max_zeta;


                let period = PERIODICITY * 2.0f32.powf(mid_zeta);
                // let period = PERIODICITY * 2.0f32.powf(zeta);
                let p = (profile_index as f32 * period) / PROFILE_BUFFER_LENGTH as f32;


                let wave_length = 2.0f32.powf(zeta);
                let wave_number = TAU / wave_length;

                let phase1 = (wave_number * (p         )) - (dispersion_relation(wave_number) * settings.elapsed_time);
                let phase2 = (wave_number * (p - period)) - (dispersion_relation(wave_number) * settings.elapsed_time);

                let weight1 = p / period;
                let weight2 = 1.0 - weight1;


                result += wave_length * spectrum(zeta, wind_speed) * (
                    // (gerstner_wave(phase1, wave_number))
                      (cubic_bump(weight1) * gerstner_wave(phase1, wave_number))
                    + (cubic_bump(weight2) * gerstner_wave(phase2, wave_number))
                );

                // result += wave_length * spectrum(zeta, wind_speed) * (
                //     (gerstner_wave(phase1, wave_number))
                //         + (gerstner_wave(phase2, wave_number))
                // );

                // result += wave_length * spectrum(zeta, wind_speed) * (
                //     gerstner_wave(phase1, wave_number)
                // );

                // result += spec * (
                //     (cubic_bump(weight1) * gerstner1)
                //     + (cubic_bump(weight2) * gerstner2)
                // );
            }
            // result /= INTEGRATION_STEPS as f32;
            // result *= (max_zeta-min_zeta);
            result *= dx;

            profile_out[zeta_index][profile_index] = result;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WATER HEIGHT RECONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Height field output: [render_z][render_x] -> (offset_x, offset_y, offset_z, 1.0)
pub type HeightBuffer = Vec<Vec<Vec4>>;

/// Normal field output: [render_z][render_x] -> (nx, ny, nz, 1.0)
pub type NormalBuffer = Vec<Vec<Vec4>>;

/// Create a new height buffer initialized to zero
pub fn new_height_buffer() -> HeightBuffer {
    vec![vec![Vec4::ZERO; RENDER_X_SIZE]; RENDER_Z_SIZE]
}

/// Create a new normal buffer initialized to zero
pub fn new_normal_buffer() -> NormalBuffer {
    vec![vec![Vec4::ZERO; RENDER_X_SIZE]; RENDER_Z_SIZE]
}

/// Sample amplitude with bilinear interpolation
fn sample_amplitude(
    amplitude_in: &AmplitudeBuffer,
    x: f32,
    z: f32,
    theta_index: usize,
    zeta_index: usize,
) -> Vec2 {
    let layer = layer_index(theta_index, zeta_index);

    // Clamp to valid range
    let x_clamped = x.clamp(0.0, (GRID_X_SIZE - 1) as f32);
    let z_clamped = z.clamp(0.0, (GRID_Z_SIZE - 1) as f32);

    let x0 = x_clamped.floor() as usize;
    let z0 = z_clamped.floor() as usize;
    let x1 = (x0 + 1).min(GRID_X_SIZE - 1);
    let z1 = (z0 + 1).min(GRID_Z_SIZE - 1);

    let fx = x_clamped.fract();
    let fz = z_clamped.fract();

    // Bilinear interpolation
    let v00 = amplitude_in[layer][z0][x0];
    let v10 = amplitude_in[layer][z0][x1];
    let v01 = amplitude_in[layer][z1][x0];
    let v11 = amplitude_in[layer][z1][x1];

    let mut result = Vec2::ZERO;
    for i in 0..2 {
        let top = v00[i] * (1.0 - fx) + v10[i] * fx;
        let bottom = v01[i] * (1.0 - fx) + v11[i] * fx;
        result[i] = top * (1.0 - fz) + bottom * fz;
    }

    result
}

/// CPU implementation of reconstruct_height compute shader.
/// Computes the height field from amplitudes and profile buffer.
pub fn reconstruct_height(
    amplitude_in: &AmplitudeBuffer,
    profile_in: &ProfileBuffer,
    height_out: &mut HeightBuffer,
    normal_out: &mut NormalBuffer,
    _settings: &WaterSettings,
    mut gizmos: Option<&mut Gizmos>,
) {
    let zeta_full_range = MAX_ZETA - MIN_ZETA;
    let direction_multiplier = 1.0 / DIRECTIONS as f32;
    let offset = vec3(-0.0, 1.0, 0.0);

    if let Some(ref mut gizmos) = gizmos {
        for zeta_index in 0..TOTAL_ZETA_NUMBERS {
            let min_zeta = MIN_ZETA + (zeta_full_range * (((zeta_index as f32) - 0.0) / (TOTAL_ZETA_NUMBERS as f32)));
            let max_zeta = MIN_ZETA + (zeta_full_range * (((zeta_index as f32) + 1.0) / (TOTAL_ZETA_NUMBERS as f32)));
            let mid_zeta = MIN_ZETA + (zeta_full_range * ((zeta_index as f32 + 0.5) / TOTAL_ZETA_NUMBERS as f32));
            let pos = vec3(0.0, (-0.05 * (zeta_index as f32 + 1.0)), 0.0) + offset;
            let period = PERIODICITY * 2.0f32.powf(mid_zeta);
            let wave_length = 2.0f32.powf(mid_zeta);

            gizmos.text(Isometry3d {
                translation: pos.into(),
                ..default()
            }, &format!("index {:?} wave_length {:?} zeta {:?}", zeta_index, wave_length, mid_zeta), 0.01, Vec2::new(0.5,0.5), Color::srgba_u8(0,255,0,255));
        }
    }

    for render_z in 0..RENDER_Z_SIZE {
        for render_x in 0..RENDER_X_SIZE {
            // Map render coordinates to grid coordinates
            let real_x = (render_x as f32 / RENDER_X_SIZE as f32) * GRID_X_SIZE as f32;
            let real_z = (render_z as f32 / RENDER_Z_SIZE as f32) * GRID_Z_SIZE as f32;

            let mut surface_offset = Vec3::ZERO;
            let mut tx = Vec3::ZERO;
            let mut tz = Vec3::ZERO;

            for zeta_index in 0..TOTAL_ZETA_NUMBERS {

                let min_zeta = MIN_ZETA + (zeta_full_range * (((zeta_index as f32) - 0.0) / (TOTAL_ZETA_NUMBERS as f32)));
                let max_zeta = MIN_ZETA + (zeta_full_range * (((zeta_index as f32) + 1.0) / (TOTAL_ZETA_NUMBERS as f32)));
                let mid_zeta = MIN_ZETA + (zeta_full_range * ((zeta_index as f32 + 0.5) / TOTAL_ZETA_NUMBERS as f32));
                let period = PERIODICITY * 2.0f32.powf(mid_zeta);
                let temp_surface_offset = surface_offset;

                // for theta_index in 0..DIRECTIONS {
                for theta_index in 0..1{
                    let theta = index_to_radians(theta_index);
                    let dir = vec3(f32::cos(theta), 0.0, f32::sin(theta));

                    // Sample amplitude at this position and direction
                    let amplitude = sample_amplitude(amplitude_in, real_x, real_z, theta_index, zeta_index);

                    // Position offset along wave direction
                    let position_offset = vec2(real_x,real_z).dot(vec2(dir.x,dir.z));

                    // Profile buffer index (wrapping)
                    let profile_coord = position_offset + DIRECTIONAL_OFFSETS[theta_index];
                    let index = modulo(
                        PROFILE_BUFFER_LENGTH as f32 * (profile_coord / period),
                        PROFILE_BUFFER_LENGTH as f32,
                    ) as usize;
                    let index = index.min(PROFILE_BUFFER_LENGTH - 1);

                    // Load profile values (scaled by direction multiplier)
                    let cos_val: Vec4 = profile_in[zeta_index][index] * direction_multiplier;

                    // Height contribution: Re(A) * cos - Im(A) * sin
                    surface_offset.y += (amplitude.x * cos_val.y) - (amplitude.y * -cos_val.x);

                    surface_offset.x += dir.x * ((amplitude.x * cos_val.x) - (amplitude.y * cos_val.y));
                    surface_offset.z += dir.y * ((amplitude.x * cos_val.x) - (amplitude.y * cos_val.y));
                    tx += dir.x * vec3(
                        ((amplitude.x * cos_val.z) - (amplitude.y * cos_val.w)),
                        ((amplitude.x * cos_val.w) - (amplitude.y * cos_val.z)),
                        0.0
                    );

                    tz += dir.y * vec3(
                        0.0,
                        ((amplitude.x * cos_val.w) - (amplitude.y * cos_val.z)),
                        ((amplitude.x * cos_val.z) - (amplitude.y * cos_val.w))
                    );
                }
                let temp_surface_offset = surface_offset-temp_surface_offset;
                if let Some(ref mut gizmos) = gizmos {
                    let pos = vec3(render_x as f32/RENDER_X_SIZE as f32, (-0.05*(zeta_index as f32+1.0)), render_z as f32/RENDER_Z_SIZE as f32) + offset;
                    // gizmos.line(pos,
                    //     pos+(temp_surface_offset/(RENDER_X_SIZE as f32 * 1.0)),
                    //     Color::srgba(1.0, render_x as f32/RENDER_X_SIZE as f32, render_z as f32/RENDER_Z_SIZE as f32, 1.0),
                    // );
                }
            }

            if let Some(ref mut gizmos) = gizmos {
                if(render_x==0&&render_z==0){
                    gizmos.text(Isometry3d {
                        translation: (offset + vec3(0.0, 1.0, 0.0)).into(),
                        ..default()
                    }, &format!("{:?} x {:?}", tx, tz), 0.01, Vec2::new(0.5,0.5), Color::srgba_u8(255,0,0,255));
                }
            }

            // Compute normal from tangent vectors
            if(tx.length()==0.0){
                tx = vec3(1.0, 0.0, 0.0);
            }
            if(tz.length()==0.0){
                tz = vec3(0.0, 0.0, 1.0);
            }
            let normal = Vec3::cross(tx, tz);
            let normal = Vec3::normalize(normal);
            // let normal_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            // let normal = if normal_len > 0.0 {
            //     [normal[0] / normal_len, normal[1] / normal_len, normal[2] / normal_len]
            // } else {
            //     [0.0, 1.0, 0.0]
            // };

            height_out[render_z][render_x] = Vec4::new(surface_offset.x, surface_offset.y, surface_offset.z, 1.0);
            normal_out[render_z][render_x] = Vec4::new(normal.x, normal.y, normal.z, 1.0);
            // normal_out[render_z][render_x] = [normal[0], normal[1], normal[2], 1.0];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE / FULL SIMULATION STEP
// ═══════════════════════════════════════════════════════════════════════════════

/// All buffers needed for CPU water simulation
pub struct CpuWaterSimulation {
    pub amplitude: AmplitudeBuffer,
    pub profile: ProfileBuffer,
    pub height: HeightBuffer,
    pub normal: NormalBuffer,
}

impl CpuWaterSimulation {
    /// Create a new CPU water simulation with all buffers initialized
    pub fn new() -> Self {
        Self {
            amplitude: new_amplitude_buffer(),
            profile: new_profile_buffer(),
            height: new_height_buffer(),
            normal: new_normal_buffer(),
        }
    }

    /// Run the water injection pass
    pub fn inject(&mut self, settings: &WaterSettings, debug_settings: &WaterDebugSettings) {
        inject_amplitude(&mut self.amplitude, &self.profile, settings, debug_settings);
    }

    /// Run the profile generation pass
    pub fn build_profile(&mut self, settings: &WaterSettings) {
        build_profile(&mut self.profile, settings);
    }

    /// Run the height reconstruction pass
    pub fn reconstruct_height(&mut self, settings: &WaterSettings) {
        reconstruct_height(
            &self.amplitude,
            &self.profile,
            &mut self.height,
            &mut self.normal,
            settings,
            None,
        );
    }

    /// Run full simulation step: profile generation, then height reconstruction
    /// (injection is separate as it's triggered by user input)
    pub fn step(&mut self, settings: &WaterSettings) {
        self.build_profile(settings);
        self.reconstruct_height(settings);
    }

    /// Get the height at a specific render coordinate
    pub fn get_height(&self, render_x: usize, render_z: usize) -> f32 {
        if render_x < RENDER_X_SIZE && render_z < RENDER_Z_SIZE {
            self.height[render_z][render_x][1]
        } else {
            0.0
        }
    }

    /// Get the surface offset at a specific render coordinate
    pub fn get_surface_offset(&self, render_x: usize, render_z: usize) -> [f32; 3] {
        if render_x < RENDER_X_SIZE && render_z < RENDER_Z_SIZE {
            [
                self.height[render_z][render_x][0],
                self.height[render_z][render_x][1],
                self.height[render_z][render_x][2],
            ]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Get the normal at a specific render coordinate
    pub fn get_normal(&self, render_x: usize, render_z: usize) -> [f32; 3] {
        if render_x < RENDER_X_SIZE && render_z < RENDER_Z_SIZE {
            [
                self.normal[render_z][render_x][0],
                self.normal[render_z][render_x][1],
                self.normal[render_z][render_x][2],
            ]
        } else {
            [0.0, 1.0, 0.0]
        }
    }
}

impl Default for CpuWaterSimulation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_to_radians() {
        // First direction should be near 0
        let r0 = index_to_radians(0);
        assert!(r0 > 0.0 && r0 < TAU / DIRECTIONS as f32);

        // Last direction should be near TAU
        let r_last = index_to_radians(DIRECTIONS - 1);
        assert!(r_last < TAU && r_last > TAU - TAU / DIRECTIONS as f32);
    }

    #[test]
    fn test_layer_index() {
        assert_eq!(layer_index(0, 0), 0);
        assert_eq!(layer_index(1, 0), 1);
        assert_eq!(layer_index(0, 1), DIRECTIONS);
        assert_eq!(layer_index(DIRECTIONS - 1, TOTAL_ZETA_NUMBERS - 1),
                   TOTAL_ZETA_NUMBERS * DIRECTIONS - 1);
    }

    #[test]
    fn test_modulo() {
        assert!((modulo(5.5, 3.0) - 2.5).abs() < 1e-6);
        assert!((modulo(-0.5, 3.0) - 2.5).abs() < 1e-6);
        assert!((modulo(3.0, 3.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_spectrum() {
        // Spectrum should be positive for reasonable inputs
        let s = spectrum(0.0, 3.0);
        assert!(s > 0.0);
    }

    #[test]
    fn test_cubic_bump() {
        // At center
        assert!((cubic_bump(0.0) - 1.0).abs() < 1e-6);
        // At edges
        assert!((cubic_bump(1.0) - 0.0).abs() < 1e-6);
        assert!((cubic_bump(-1.0) - 0.0).abs() < 1e-6);
        // Outside
        assert!((cubic_bump(1.5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gerstner_wave() {
        let g = gerstner_wave(0.0, 1.0);
        // At phase 0: sin(0)=0, cos(0)=1
        assert!((g[0] - 0.0).abs() < 1e-6); // -sin(0) = 0
        assert!((g[1] - 1.0).abs() < 1e-6); // cos(0) = 1
    }

    #[test]
    fn test_simulation_creation() {
        let sim = CpuWaterSimulation::new();
        assert_eq!(sim.amplitude.len(), TOTAL_ZETA_NUMBERS * DIRECTIONS);
        assert_eq!(sim.profile.len(), TOTAL_ZETA_NUMBERS);
        assert_eq!(sim.height.len(), RENDER_Z_SIZE);
        assert_eq!(sim.normal.len(), RENDER_Z_SIZE);
    }
}
