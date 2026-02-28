// Build the 1-D profile buffer Psi(p)
// Precomputes cos(offset - phase) for height reconstruction
// Dispatched with workgroups (ceil(PROFILE_BUFFER_LENGTH/64), 1, 1)

#import "water_common.wgsl"::{
    PROFILE_BUFFER_LENGTH, TOTAL_WAVE_NUMBERS,
    GRAVITY, TIME_MULT, TAU, WaterSettings
}

@group(0) @binding(0) var profile_out: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var<uniform> settings: WaterSettings;

@compute @workgroup_size(128, 1, 1)
fn build_profile(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p = global_id.x;

    if (p >= PROFILE_BUFFER_LENGTH) {
        return;
    }

    // For each wave number, compute the profile value at this index
    for (var wave_index: u32 = 0u; wave_index < TOTAL_WAVE_NUMBERS; wave_index++) {
        // Wave number (matching CPU WAVE_NUMBERS)
        let wave_number = 0.25;

        // Angular frequency: omega = sqrt(g * k)
        let angular_frequency = sqrt(GRAVITY * wave_number);

        // Time-dependent phase coordinate
        let phase_coordinate = settings.elapsed_time * TIME_MULT * angular_frequency;

        // Spatial offset within the profile buffer (0 to TAU)
        let offset = f32(p) / f32(PROFILE_BUFFER_LENGTH) * TAU;

        // Compute the profile value: cos(offset - phase)
        let profile_value = cos(offset - phase_coordinate);

        // Store in profile texture (2D texture: x = profile index, y = wave index)
        textureStore(profile_out, vec2<i32>(i32(p), i32(wave_index)), vec4<f32>(profile_value, 0.0, 0.0, 1.0));
    }
}
