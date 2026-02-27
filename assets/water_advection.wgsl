// Semi-Lagrangian advection of wave amplitudes
// Dispatched with workgroups (ceil(GRID_X/8), ceil(GRID_Z/8), DIRECTIONS)
// Each invocation handles one (x, z) cell for one direction

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

@group(0) @binding(0) var amplitude_in: texture_2d_array<f32>;
@group(0) @binding(1) var amplitude_sampler: sampler;
@group(0) @binding(2) var amplitude_out: texture_storage_2d_array<rg32float, write>;
@group(0) @binding(3) var<uniform> settings: WaterSettings;

fn index_to_radians(i: u32) -> f32 {
    return f32(i) * (TAU / f32(DIRECTIONS));
}

fn layer_index(theta_index: u32, wave_index: u32) -> i32 {
    return i32(wave_index * DIRECTIONS + theta_index);
}

// Hardware bilinear sampling using sampler
fn sample_amplitude(x: f32, z: f32, theta_index: u32, wave_index: u32) -> vec2<f32> {
    // Boundary check - return zero outside valid range
    if (x < 1.0 || x >= f32(GRID_X_SIZE - 1u) || z < 1.0 || z >= f32(GRID_Z_SIZE - 1u)) {
        return vec2<f32>(0.0, 0.0);
    }

    // Convert to normalized UV coordinates (0-1)
    // Add 0.5 to sample at texel centers
    let uv = vec2<f32>((x + 0.5) / f32(GRID_X_SIZE), (z + 0.5) / f32(GRID_Z_SIZE));
    let layer = layer_index(theta_index, wave_index);

    return textureSampleLevel(amplitude_in, amplitude_sampler, uv, layer, 0.0).xy;
}

fn wrap_direction(d: i32) -> u32 {
    return u32((d % i32(DIRECTIONS) + i32(DIRECTIONS)) % i32(DIRECTIONS));
}

@compute @workgroup_size(8, 8, 1)
fn advect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let z = global_id.y;
    let theta_index = global_id.z;

    // Bounds check
    if (x >= GRID_X_SIZE || z >= GRID_Z_SIZE || theta_index >= DIRECTIONS) {
        return;
    }

    let theta = index_to_radians(theta_index);

    for (var wave_index: u32 = 0u; wave_index < TOTAL_WAVE_NUMBERS; wave_index++) {
        // Wave number = 1.0 for now (matching CPU WAVE_NUMBERS[0])
        let wave_number = 1.0;

        // Phase speed and group speed
        let phase_speed = sqrt(GRAVITY / wave_number);
        let group_speed = phase_speed / 2.0;

        // Displacement based on group velocity
        let dx = cos(theta) * group_speed * settings.delta_time * TIME_MULT;
        let dz = sin(theta) * group_speed * settings.delta_time * TIME_MULT;

        // Previous position (semi-Lagrangian)
        let prev_x = f32(x) - dx;
        let prev_z = f32(z) - dz;

        // Sample previous amplitude at current direction
        let prev = sample_amplitude(prev_x, prev_z, theta_index, wave_index);

        // Directional blur - sample adjacent angles with weights
        // This smooths the amplitude across nearby propagation directions
        let theta_m1 = wrap_direction(i32(theta_index) - 1);
        let theta_p1 = wrap_direction(i32(theta_index) + 1);

        let mag_m1 = length(sample_amplitude(prev_x, prev_z, theta_m1, wave_index));
        let mag_0  = length(sample_amplitude(prev_x, prev_z, theta_index, wave_index));
        let mag_p1 = length(sample_amplitude(prev_x, prev_z, theta_p1, wave_index));

        // Weighted average of magnitudes
        let new_magnitude = mag_m1 * 0.25 + mag_0 * 0.50 + mag_p1 * 0.25;

        // Preserve the phase direction from the center sample
        let normalized = normalize(select(vec2<f32>(1.0, 0.0), prev, length(prev) > 0.0001));
        let new_amplitude = normalized * new_magnitude;

        // Write to output
        let layer = layer_index(theta_index, wave_index);
        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer, vec4<f32>(new_amplitude, 0.0, 1.0));
    }
}
