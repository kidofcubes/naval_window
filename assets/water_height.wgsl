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

@compute @workgroup_size(8, 8, 1)
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
            let wave_number = 1.0;

            // Sample amplitude at this position and direction
            let amplitude = sample_amplitude(real_x, real_z, theta_index, wave_index);

            // Position offset along wave direction
            let position_offset = dot(vec2<f32>(real_x, real_z), dir);

            // Profile buffer index (wrapping)
            let profile_coord = (wave_number * position_offset / TAU) * f32(PROFILE_BUFFER_LENGTH);
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
