// Debug amplitude injection compute shader
// Injects phase-coherent waves in a circular region for debugging
// Dispatched with workgroups (ceil(GRID_X/8), ceil(GRID_Z/8), DIRECTIONS)

#import "water_common.wgsl"::{
    GRID_X_SIZE, GRID_Z_SIZE, RENDER_X_SIZE, RENDER_Z_SIZE,
    DIRECTIONS, TOTAL_ZETA_NUMBERS, PROFILE_BUFFER_LENGTH,
    ZETA_MIN, ZETA_MAX, DIRECTIONAL_OFFSETS,
    GRAVITY, TIME_MULT, TAU, WaterSettings, WaterDebugSettings, VoxelSettings, WaterWaveGroupSpeeds,
    index_to_radians, layer_index
}

@group(0) @binding(0) var amplitude_out: texture_storage_2d_array<rg32float, read_write>;
@group(0) @binding(1) var<uniform> settings: WaterSettings;
@group(0) @binding(2) var<uniform> debug_settings: WaterDebugSettings;

@compute @workgroup_size(32, 32, 1)
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
    let dist = distance(pos, vec2(debug_settings.spawn_center_x, debug_settings.spawn_center_z));

    // Skip if outside injection radius
    if (dist > debug_settings.radius) {
        return;
    }

    let theta = index_to_radians(theta_index);
    let dir = vec2<f32>(cos(theta), sin(theta));

    for (var zeta_index: u32 = 0u; zeta_index < TOTAL_ZETA_NUMBERS; zeta_index++) {
//        let wave_number = WAVE_NUMBERS[zeta_index];
//        let angular_frequency = sqrt(GRAVITY * wave_number);

        // Position offset along wave direction
        let position_offset = dot(pos, dir);

        // Phase coordinate: counteract both position and time to create stationary source
        let phase_coordinate = (1.0 * position_offset) - (settings.elapsed_time * TIME_MULT);

        // Create amplitude with correct phase
        let amplitude = vec2<f32>(
            debug_settings.scale * cos(-phase_coordinate),
            debug_settings.scale * sin(-phase_coordinate)
        );

        let layer = layer_index(theta_index, zeta_index);
//        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer, vec4<f32>(-amplitude, 0.0, 1.0));
        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer, vec4<f32>(debug_settings.scale, 0.0, 0.0, 1.0));
    }
}
