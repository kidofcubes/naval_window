// Debug amplitude injection compute shader
// Injects phase-coherent waves in a circular region for debugging
// Dispatched with workgroups (ceil(GRID_X/8), ceil(GRID_Z/8), DIRECTIONS)

const GRID_X_SIZE: u32 = 128u;
const GRID_Z_SIZE: u32 = 128u;
const DIRECTIONS: u32 = 64u;
const TOTAL_WAVE_NUMBERS: u32 = 1u;
const GRAVITY: f32 = 9.81;
const TIME_MULT: f32 = 3.0;
const TAU: f32 = 6.283185307179586;

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
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var amplitude_out: texture_storage_2d_array<rg32float, read_write>;
@group(0) @binding(1) var<uniform> settings: WaterSettings;
@group(0) @binding(2) var<uniform> debug_settings: WaterDebugSettings;

fn index_to_radians(i: u32) -> f32 {
    return f32(i) * (TAU / f32(DIRECTIONS));
}

fn layer_index(theta_index: u32, wave_index: u32) -> i32 {
    return i32(wave_index * DIRECTIONS + theta_index);
}

@compute @workgroup_size(8, 8, 1)
fn inject_amplitude(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let z = global_id.y;
    let theta_index = global_id.z;

    // Bounds check
    if (x >= GRID_X_SIZE || z >= GRID_Z_SIZE || theta_index >= DIRECTIONS) {
        return;
    }

    // Skip if injection is not active
    if (debug_settings.inject < 0.5) {
        return;
    }

    let pos = vec2<f32>(f32(x), f32(z));
    let dist = distance(pos, vec2(debug_settings.spawn_center_x,debug_settings.spawn_center_z));

    // Skip if outside injection radius
    if (dist > debug_settings.radius) {
        return;
    }

    let theta = index_to_radians(theta_index);
    let dir = vec2<f32>(cos(theta), sin(theta));

    for (var wave_index: u32 = 0u; wave_index < TOTAL_WAVE_NUMBERS; wave_index++) {
        let wave_number = 1.0;
        let angular_frequency = sqrt(GRAVITY * wave_number);

        // Position offset along wave direction
        let position_offset = dot(pos, dir);

        // Phase coordinate: counteract both position and time to create stationary source
        let phase_coordinate = (wave_number * position_offset) - (settings.elapsed_time * TIME_MULT * angular_frequency);

        // Create amplitude with correct phase
        let amplitude = vec2<f32>(
            debug_settings.scale * cos(-phase_coordinate),
            debug_settings.scale * sin(-phase_coordinate)
        );

        let layer = layer_index(theta_index, wave_index);
        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer, vec4<f32>(amplitude, 0.0, 1.0));
//        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer, vec4<f32>(1.0, 0.0, 0.0, 1.0));
    }
}
