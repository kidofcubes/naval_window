// Height field reconstruction
// Computes eta(x, y) = sum over all directions of A(x,y,d) * profile lookup
// Dispatched with workgroups (ceil(RENDER_X/8), ceil(RENDER_Z/8), 1)

#import "water_common.wgsl"::{
    GRID_X_SIZE, GRID_Z_SIZE, RENDER_X_SIZE, RENDER_Z_SIZE,
    DIRECTIONS, TOTAL_WAVE_NUMBERS, PROFILE_BUFFER_LENGTH,
    TAU, WaterSettings, index_to_radians, layer_index
}

@group(0) @binding(0) var amplitude_in: texture_2d_array<f32>;
@group(0) @binding(1) var amplitude_sampler: sampler;
@group(0) @binding(2) var profile_in: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var height_out: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var<uniform> settings: WaterSettings;

// Hardware bilinear sample from amplitude texture array
fn sample_amplitude(x: f32, z: f32, theta_index: u32, wave_index: u32) -> vec2<f32> {
    // Convert to normalized UV coordinates (0-1)
    // Add 0.5 to sample at texel centers
    let uv = vec2<f32>((x + 0.5) / f32(GRID_X_SIZE), (z + 0.5) / f32(GRID_Z_SIZE));
    let layer = layer_index(theta_index, wave_index);

    return textureSampleLevel(amplitude_in, amplitude_sampler, uv, layer, 0.0).xy;
}

const offsets = array<f32, 64>(4.00209964, 0.62348099, 0.61843034, 6.26698587, 6.17683248,
                                   2.22353755, 5.9510902 , 5.170589  , 0.57009176, 3.66137778,
                                   5.20795387, 2.82310305, 3.24363374, 1.04382024, 1.03452628,
                                   0.78082669, 4.60312318, 5.89416825, 5.94252555, 3.28285452,
                                   5.55188924, 4.59259198, 0.43568562, 6.22158418, 3.59665422,
                                   1.21573896, 0.43210179, 0.66657046, 0.86753942, 1.99964832,
                                   2.31027937, 2.39669034, 2.16768309, 5.3520594 , 6.10021014,
                                   0.64603184, 5.39703777, 2.73254197, 1.32722682, 2.01059004,
                                   5.73428959, 0.4953346 , 0.66897448, 0.95173938, 3.81079024,
                                   4.85437434, 5.09746399, 5.38410691, 6.28016522, 5.47544104,
                                   4.71553984, 6.25085506, 2.36433207, 0.68478357, 1.46457622,
                                   1.10174288, 6.22337483, 0.74105602, 1.31046907, 1.6523683 ,
                                   1.41188214, 5.51205444, 2.1722495 , 1.52310835);

@compute @workgroup_size(32, 32, 1)
fn reconstruct_height(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let render_x = global_id.x;
    let render_z = global_id.y;

    if (render_x >= RENDER_X_SIZE || render_z >= RENDER_Z_SIZE) {
        return;
    }

    // Map render coordinates to grid coordinates
    let real_x = f32(render_x) / f32(RENDER_X_SIZE) * f32(GRID_X_SIZE);
    let real_z = f32(render_z) / f32(RENDER_Z_SIZE) * f32(GRID_Z_SIZE);

    var height = 0.0;

    for (var theta_index: u32 = 0u; theta_index < DIRECTIONS; theta_index++) {
        let theta = index_to_radians(theta_index);
        let dir = vec2<f32>(cos(theta), sin(theta));

        for (var wave_index: u32 = 0u; wave_index < TOTAL_WAVE_NUMBERS; wave_index++) {
            let wave_number = 0.25;

            // Sample amplitude at this position and direction
            let amplitude = sample_amplitude(real_x, real_z, theta_index, wave_index);

            // Position offset along wave direction
            let position_offset = dot(vec2<f32>(real_x, real_z), dir);

            // Profile buffer index (wrapping)
            let profile_coord = ((wave_number * position_offset / TAU) + (offsets[theta_index]/4.0)) * f32(PROFILE_BUFFER_LENGTH);
            let profile_index = i32(profile_coord) % i32(PROFILE_BUFFER_LENGTH);
            let wrapped_index = select(profile_index, profile_index + i32(PROFILE_BUFFER_LENGTH), profile_index < 0);

            // Index for 90-degree phase shift (sin from cos)
            let sin_index = (wrapped_index + i32((3u * PROFILE_BUFFER_LENGTH) / 4u)) % i32(PROFILE_BUFFER_LENGTH);

            // Load profile values (cos and sin components)
            let cos_val = textureLoad(profile_in, vec2<i32>(wrapped_index, i32(wave_index))).x;
            let sin_val = textureLoad(profile_in, vec2<i32>(sin_index, i32(wave_index))).x;

            // Height contribution: Re(A) * cos - Im(A) * sin
            // This properly handles the complex amplitude with phase
            height += amplitude.x * cos_val - amplitude.y * sin_val;
        }
    }

    // Scale and store
    height *= settings.height_scale;
    textureStore(height_out, vec2<i32>(i32(render_x), i32(render_z)), vec4<f32>(height, 0.0, 0.0, 1.0));
}
