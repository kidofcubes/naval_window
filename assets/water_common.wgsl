//#define_import_path water_common

// Grid dimensions
const GRID_X_SIZE: u32 = 128u;
const GRID_Z_SIZE: u32 = 128u;
const RENDER_X_SIZE: u32 = 256u;
const RENDER_Z_SIZE: u32 = 256u;
const DIRECTIONS: u32 = 64u;
const TOTAL_WAVE_NUMBERS: u32 = 1u;
const PROFILE_BUFFER_LENGTH: u32 = 2048u;

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

// Helper functions
fn index_to_radians(i: u32) -> f32 {
    return f32(i) * (TAU / f32(DIRECTIONS));
}

fn layer_index(theta_index: u32, wave_index: u32) -> i32 {
    return i32(wave_index * DIRECTIONS + theta_index);
}
