// Debug amplitude injection compute shader
// Injects phase-coherent waves in a circular region for debugging
// Dispatched with workgroups (ceil(GRID_X/32), ceil(GRID_Z/32), DIRECTIONS)

#import "water_common.wgsl"::{
    GRID_X_SIZE, GRID_Z_SIZE, RENDER_X_SIZE, RENDER_Z_SIZE,
    DIRECTIONS, TOTAL_ZETA_NUMBERS, PROFILE_BUFFER_LENGTH,
    PERIODICITY, ZETA_MIN, ZETA_MAX, DIRECTIONAL_OFFSETS,
    GRAVITY, TIME_MULT, TAU, WaterSettings, WaterDebugSettings, VoxelSettings, PreComputedWaveData,
    index_to_radians, layer_index
}

@group(0) @binding(0) var amplitude_out: texture_storage_2d_array<r32float, read_write>;
@group(0) @binding(1) var<uniform> settings: WaterSettings;
@group(0) @binding(2) var<uniform> debug_settings: WaterDebugSettings;
@group(0) @binding(3) var<uniform> pre_computed_wave_data: PreComputedWaveData;
@group(0) @binding(4) var profile_in: texture_storage_2d<rgba32float, read>;

const PI: f32 = 3.141592653589793;

fn modulo(x: f32, n: f32) -> f32 {
    return x - floor(x / n) * n;
}

// Mirrors cpu_debug_impl::get_offset_for_position.
// Returns vec2(horizontal_disp, vertical_height) from the profile lookup.
fn get_offset_for_position(
    grid_pos: vec2<f32>,
    zeta_index: u32,
    theta_index: u32,
) -> vec2<f32> {
    let theta = index_to_radians(theta_index);
    let dir = vec3<f32>(cos(theta), 0.0, sin(theta));

    let zeta_full_range = ZETA_MAX - ZETA_MIN;
    let mid_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) + 0.5) / f32(TOTAL_ZETA_NUMBERS)));
    let period = PERIODICITY * pow(2.0, mid_zeta);

    let position_offset = dot(grid_pos, vec2<f32>(dir.x, dir.z));
    let profile_coord = position_offset + DIRECTIONAL_OFFSETS[theta_index];

    let raw_index = modulo(
        f32(PROFILE_BUFFER_LENGTH) * (profile_coord / period),
        f32(PROFILE_BUFFER_LENGTH),
    );
    let index = min(u32(raw_index), PROFILE_BUFFER_LENGTH - 1u);

    let profile_value = textureLoad(profile_in, vec2<i32>(i32(index), i32(zeta_index)));

    var offset = vec2<f32>(0.0, 0.0);
    let amplitude = vec2(1.0,0.0);
    let cos_val = profile_value;

    offset.y += (amplitude.x * cos_val.y) - (amplitude.y * -cos_val.x); //vertical

    offset.x += (amplitude.x * cos_val.x) - (amplitude.y *  cos_val.y); //horizontal

    return offset;
}

@compute @workgroup_size(32, 32, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let z = global_id.y;
    let theta_index = global_id.z;

    // Bounds check
    if (x >= GRID_X_SIZE || z >= GRID_Z_SIZE || theta_index >= DIRECTIONS) {
        return;
    }

    // Skip if injection is not active
//    if (debug_settings.inject < 0.5) {
//        return;
//    }

    let pos_x = f32(x);
    let pos_z = f32(z);
    let dist = distance(
        vec2<f32>(pos_x, pos_z),
        vec2<f32>(debug_settings.spawn_center_x, debug_settings.spawn_center_z),
    );

    // Skip if outside injection radius
//    if (dist > debug_settings.radius) {
//        return;
//    }

    let theta = index_to_radians(theta_index);
    let dir = vec3<f32>(cos(theta), 0.0, sin(theta));
    let zeta_full_range = ZETA_MAX - ZETA_MIN;

    for (var zeta_index: u32 = 0u; zeta_index < TOTAL_ZETA_NUMBERS; zeta_index++) {
        let min_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) - 0.0) / f32(TOTAL_ZETA_NUMBERS)));
        let max_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) + 1.0) / f32(TOTAL_ZETA_NUMBERS)));
        let mid_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) + 0.5) / f32(TOTAL_ZETA_NUMBERS)));
        let period = PERIODICITY * pow(2.0, mid_zeta);

        let wave_length0 = pow(2.0, min_zeta);
        let wave_number0 = TAU / wave_length0;

        let wave_length1 = pow(2.0, mid_zeta);
        let wave_number1 = TAU / wave_length1;

        let wave_length2 = pow(2.0, max_zeta);
        let wave_number2 = TAU / wave_length2;

        // Position offset along wave direction
        let position_offset = dot(vec2<f32>(pos_x, pos_z), vec2<f32>(dir.x, dir.z));

        // profile coord from height reconstruction
        let profile_coord = position_offset + DIRECTIONAL_OFFSETS[theta_index];

//        let spatial_phase =
//            (wave_number1 * profile_coord)
//          - (sqrt(GRAVITY * wave_number1) * settings.elapsed_time);
//        var phase_coordinate = spatial_phase;

        let raw_index = modulo(
            f32(PROFILE_BUFFER_LENGTH) * (profile_coord / period),
            f32(PROFILE_BUFFER_LENGTH),
        );
        let index = min(u32(raw_index), PROFILE_BUFFER_LENGTH - 1u);

//        if (debug_settings.radius > 0.0) {
            let orig_offset = get_offset_for_position(vec2(pos_x,pos_z), zeta_index, theta_index);
            let orig_angle = atan2(orig_offset.y, orig_offset.x);
            let phase_coordinate = orig_angle - PI / 2.0;
//            phase_coordinate = 0.0;
//        }

        // amplitude = (cos(-phase), sin(-phase)) * scale
        // r32float storage can only hold the real (cos) component.
        let layer = layer_index(theta_index, zeta_index);

        let orig_amplitude = textureLoad(amplitude_out, vec2<i32>(i32(x), i32(z)), layer).xy;
        var amplitude = orig_amplitude;



        var normalized_amplitude = normalize(orig_amplitude);
        if(length(orig_amplitude)==0.000){
            normalized_amplitude = vec2<f32>(1.0, 0.0);
        }
        let relax_constant = 1.0/(600.0 * (min_zeta / mid_zeta));
//        let relax_constant = 1.0;
//        let relax_constant = 600000.0;
        let orig_length = length(orig_amplitude);
//        amplitude = orig_amplitude + vec2(1.0,0.0);
        let difference = ((pre_computed_wave_data.jonswap_spectrum_amplitudes[layer].x) - orig_length);
//        let difference = 0.0;
//        amplitude = orig_amplitude + (normalized_amplitude * max(0.0, difference * relax_constant));
        amplitude = (normalized_amplitude * pre_computed_wave_data.jonswap_spectrum_amplitudes[layer].x);
//        amplitude = orig_amplitude + (normalized_amplitude * max(0.0, (10.0 - orig_length) / relax_constant));
//        amplitude = orig_amplitude + (normalized_amplitude * 0.0000000000000001);
//        amplitude = vec2(10.0,0.0);


        if (dist <= debug_settings.radius && debug_settings.inject > 0.5) {
            amplitude = vec2<f32>(
                cos(-phase_coordinate),
                sin(-phase_coordinate),
            ) * debug_settings.scale;
        }


        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer,
                     vec4<f32>(amplitude.x, amplitude.y, 0.0, 1.0));
    }
}
