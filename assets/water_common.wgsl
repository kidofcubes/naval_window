//#define_import_path water_common

// Grid dimensions
const GRID_X_SIZE: u32 = 1024u;
const GRID_Z_SIZE: u32 = 1024u;
const RENDER_X_SIZE: u32 = 1024u;
const RENDER_Z_SIZE: u32 = 1024u;
const DIRECTIONS: u32 = 64u;
const TOTAL_WAVE_NUMBERS: u32 = 1u;
const PROFILE_BUFFER_LENGTH: u32 = 2048u;

// Wave parameters
const ZETA_MIN: f32 = -5.059;
const ZETA_MAX: f32 =  3.322;
//const WAVE_NUMBERS = array<f32, 1>(1.0);
const WAVE_NUMBERS = array<f32, 1>(1.0);
const DIRECTIONAL_OFFSETS = array<f32, 64>(
    4.00209964, 0.62348099, 0.61843034, 6.26698587,
    6.17683248, 2.22353755, 5.95109020, 5.17058900,
    0.57009176, 3.66137778, 5.20795387, 2.82310305,
    3.24363374, 1.04382024, 1.03452628, 0.78082669,
    4.60312318, 5.89416825, 5.94252555, 3.28285452,
    5.55188924, 4.59259198, 0.43568562, 6.22158418,
    3.59665422, 1.21573896, 0.43210179, 0.66657046,
    0.86753942, 1.99964832, 2.31027937, 2.39669034,
    2.16768309, 5.35205940, 6.10021014, 0.64603184,
    5.39703777, 2.73254197, 1.32722682, 2.01059004,
    5.73428959, 0.49533460, 0.66897448, 0.95173938,
    3.81079024, 4.85437434, 5.09746399, 5.38410691,
    6.28016522, 5.47544104, 4.71553984, 6.25085506,
    2.36433207, 0.68478357, 1.46457622, 1.10174288,
    6.22337483, 0.74105602, 1.31046907, 1.65236830,
    1.41188214, 5.51205444, 2.17224950, 1.52310835
);


// Physical constants
const GRAVITY: f32 = 9.81;
const TIME_MULT: f32 = 3.0;
const TAU: f32 = 6.283185307179586;

// Uniform structs
struct WaterSettings {
    height_scale: f32,
    time_scale: f32,
    delta_time: f32,
    elapsed_time: f32,
}

struct WaterDebugSettings {
    spawn_center_x: f32,
    spawn_center_z: f32,
    radius: f32,
    scale: f32,
    inject: f32,
    _padding_a: f32,
    _padding_b: f32,
    _padding_c: f32,
}

struct VoxelSettings {
    water_to_voxel: mat4x4<f32>,  // Transform water grid to voxel
    voxel_size: f32,
    grid_dims: vec3<u32>,
}

// Helper functions
fn index_to_radians(i: u32) -> f32 {
    return f32(i) * (TAU / f32(DIRECTIONS));
}

fn layer_index(theta_index: u32, wave_index: u32) -> i32 {
    return i32(wave_index * DIRECTIONS + theta_index);
}
