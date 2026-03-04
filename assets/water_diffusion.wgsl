// Directional diffusion of wave amplitudes
// Dispatched with workgroups (ceil(GRID_X/32), ceil(GRID_Z/32), DIRECTIONS)
// Each invocation handles one (x, z) cell for one direction
// This pass spreads energy between adjacent propagation directions

#import "water_common.wgsl"::{
    GRID_X_SIZE, GRID_Z_SIZE, RENDER_X_SIZE, RENDER_Z_SIZE,
    DIRECTIONS, TOTAL_ZETA_NUMBERS, PROFILE_BUFFER_LENGTH,
    ZETA_MIN, ZETA_MAX, DIRECTIONAL_OFFSETS,
    GRAVITY, TIME_MULT, TAU, WaterSettings, WaterDebugSettings, VoxelSettings, WaterWaveGroupSpeeds,
    index_to_radians, layer_index
}

@group(0) @binding(0) var amplitude_in: texture_2d_array<f32>;
@group(0) @binding(1) var amplitude_sampler: sampler;
@group(0) @binding(2) var amplitude_out: texture_storage_2d_array<rg32float, write>;
@group(0) @binding(3) var<uniform> settings: WaterSettings;
@group(0) @binding(4) var<uniform> water_wave_group_speeds: WaterWaveGroupSpeeds;

fn sample_amplitude(x: u32, z: u32, theta_index: u32, wave_index: u32) -> vec2<f32> {
//    if (x < 1.0 || x >= f32(GRID_X_SIZE - 1u) || z < 1.0 || z >= f32(GRID_Z_SIZE - 1u)) {
//        return vec2<f32>(0.0, 0.0);
//    }
    let uv = vec2<f32>((f32(x) + 0.5) / f32(GRID_X_SIZE), (f32(z) + 0.5) / f32(GRID_Z_SIZE));
    let layer = layer_index(theta_index, wave_index);
    return textureSampleLevel(amplitude_in, amplitude_sampler, uv, layer, 0.0).xy;
}

fn wrap_direction(d: i32) -> u32 {
    return u32((d % i32(DIRECTIONS) + i32(DIRECTIONS)) % i32(DIRECTIONS));
}

@compute @workgroup_size(32, 32, 1)
fn diffuse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let z = global_id.y;
    let theta_index = global_id.z;

    // Bounds check
    if (x >= GRID_X_SIZE || z >= GRID_Z_SIZE || theta_index >= DIRECTIONS) {
        return;
    }

    let theta = index_to_radians(theta_index);

    for (var zeta_index: u32 = 0u; zeta_index < TOTAL_ZETA_NUMBERS; zeta_index++) {
        let layer = layer_index(theta_index, zeta_index);

        // Adjacent directions
        let theta_m1 = wrap_direction(i32(theta_index) - 1);
        let theta_p1 = wrap_direction(i32(theta_index) + 1);

        // Diffusion coefficient
        let gamma = 2.0 * 0.025 * (water_wave_group_speeds.wave_group_speeds[zeta_index].x) * (settings.delta_time * TIME_MULT) * 0.5;

        // Sample magnitudes from adjacent directions at current position
        let amp_m1 = sample_amplitude(x, z, theta_m1, zeta_index);
        let amp_0  = sample_amplitude(x, z, theta_index, zeta_index);
        let amp_p1 = sample_amplitude(x, z, theta_p1, zeta_index);

        let mag_m1 = length(amp_m1);
        let mag_0  = length(amp_0);
        let mag_p1 = length(amp_p1);

        // Weighted average of magnitudes
        let new_magnitude = (1.0 - gamma) * mag_0 + gamma * 0.5 * (mag_m1 + mag_p1);

        // Preserve the phase direction from the center sample
        let normalized = normalize(select(vec2<f32>(1.0, 0.0), amp_0, mag_0 > 0.0001));
        let new_amplitude = normalized * new_magnitude;

        // Write to output
        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer, vec4<f32>(new_amplitude, 0.0, 1.0));
    }
}
