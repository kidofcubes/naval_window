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

@group(0) @binding(0) var amplitude_out: texture_storage_2d_array<r32float, read_write>;
@group(0) @binding(1) var<uniform> settings: WaterSettings;
@group(0) @binding(2) var<uniform> debug_settings: WaterDebugSettings;

@compute @workgroup_size(32, 32, 1)
fn inject_amplitude(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let z = global_id.y;
    let theta_index = global_id.z;
//    let theta_index: u32 = 0;

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
    let dir = vec3<f32>(cos(theta), 0.0, sin(theta));
    let zeta_full_range = ZETA_MAX - ZETA_MIN;

    for (var zeta_index: u32 = 0u; zeta_index < TOTAL_ZETA_NUMBERS; zeta_index++) {
        let min_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) - 0.0) / f32(TOTAL_ZETA_NUMBERS)));
        let max_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) + 1.0) / f32(TOTAL_ZETA_NUMBERS)));


        let zeta = min_zeta + ((max_zeta - min_zeta) * ((f32(0.0) + 0.5) / f32(1.0)));
//        let zeta = max_zeta; //works if both is max_zeta
//        let period: f32 = f32(2.0) * pow(2,max_zeta);

        let waveLength0 = pow(2, min_zeta);
        let wave_number0 = TAU / waveLength0;

        let waveLength1 = pow(2, zeta);
        let wave_number1 = TAU / waveLength1;

        let waveLength2 = pow(2, max_zeta);
        let wave_number2 = TAU / waveLength2;
//        let wave_number = TAU / waveLength2;
//        let wave_number = WAVE_NUMBERS[zeta_index];
//        let angular_frequency = sqrt(GRAVITY * wave_number);

        // Position offset along wave direction
        let position_offset = dot(vec2<f32>(f32(x), f32(z)), vec2<f32>(dir.x, dir.z));



        let spatial_phase =
//                           (wave_number2 * ((position_offset+(settings.elapsed_time*0.0)) + (DIRECTIONAL_OFFSETS[theta_index]/1.0)))
                           (wave_number2 * ((position_offset+(settings.elapsed_time*0.0)) + (DIRECTIONAL_OFFSETS[theta_index]/1.0)))
//                         - (wave_number2 * ((position_offset+(settings.elapsed_time*0.0)) + (DIRECTIONAL_OFFSETS[theta_index]/1.0)))
                         - (sqrt(GRAVITY * wave_number2) * ((settings.elapsed_time)))
//                         - (sqrt(GRAVITY * wave_number2) * ((settings.elapsed_time)))
//                         - (sqrt(GRAVITY * wave_number2) * sqrt((settings.elapsed_time)))
//                         - (sqrt(GRAVITY * wave_number2) * ((settings.elapsed_time)))

                         ;


        let phase_coordinate = spatial_phase;


        // Create amplitude with correct phase
        let amplitude = vec2<f32>(
            debug_settings.scale * cos(-phase_coordinate),
            debug_settings.scale * sin(-phase_coordinate)
        );

        let layer = layer_index(theta_index, zeta_index);
//        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer, vec4<f32>(amplitude, 0.0, 1.0));
        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer, vec4<f32>(debug_settings.scale, 0.0, 0.0, 1.0));
    }
}
