// Height field reconstruction
// Computes eta(x, y) = sum over all directions of A(x,y,d) * profile lookup
// Dispatched with workgroups (ceil(RENDER_X/8), ceil(RENDER_Z/8), 1)

#import "water_common.wgsl"::{
    GRID_X_SIZE, GRID_Z_SIZE, RENDER_X_SIZE, RENDER_Z_SIZE,
    DIRECTIONS, TOTAL_ZETA_NUMBERS, PROFILE_BUFFER_LENGTH,
    PERIODICITY, ZETA_MIN, ZETA_MAX, DIRECTIONAL_OFFSETS,
    GRAVITY, TIME_MULT, TAU, WaterSettings, WaterDebugSettings, VoxelSettings, WaterWaveGroupSpeeds,
    index_to_radians, layer_index
}

@group(0) @binding(0) var amplitude_in: texture_2d_array<f32>;
@group(0) @binding(1) var amplitude_sampler: sampler;
@group(0) @binding(2) var profile_in: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var height_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var normal_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<uniform> settings: WaterSettings;

// Hardware bilinear sample from amplitude texture array
fn sample_amplitude(x: f32, z: f32, theta_index: u32, zeta_index: u32) -> f32 {
    // Convert to normalized UV coordinates (0-1)
    // Add 0.5 to sample at texel centers
    let uv = vec2<f32>((x + 0.5) / f32(GRID_X_SIZE), (z + 0.5) / f32(GRID_Z_SIZE));
    let layer = layer_index(theta_index, zeta_index);

    return textureSampleLevel(amplitude_in, amplitude_sampler, uv, layer, 0.0).x;
}


fn modulo(x: f32, n: f32) -> f32{
    return x - floor(x / n) * n;
}

//fn modulo(x: i32, n: i32) -> i32{
//    return x - i32(floor(f32(x) / f32(n))) * n;
//}

@compute @workgroup_size(32, 32, 1)
fn reconstruct_height(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let render_x = global_id.x;
    let render_z = global_id.y;

    if (render_x >= RENDER_X_SIZE || render_z >= RENDER_Z_SIZE) {
        return;
    }

    // Map render coordinates to grid coordinates
    let real_x = (f32(render_x) / f32(RENDER_X_SIZE)) * f32(GRID_X_SIZE);
    let real_z = (f32(render_z) / f32(RENDER_Z_SIZE)) * f32(GRID_Z_SIZE);

    var surface_offset = vec3<f32>(0.0, 0.0, 0.0);

    var tx = vec3<f32>(0.0, 0.0, 0.0);
    var tz = vec3<f32>(0.0, 0.0, 0.0);

    let direction_multiplier = 1.0/f32(DIRECTIONS);
//    let direction_multiplier = 10.0;

    var amplitude_sum = vec2<f32>(0.0, 0.0);

    let zeta_full_range = ZETA_MAX - ZETA_MIN;

    for (var zeta_index: u32 = 0u; zeta_index < TOTAL_ZETA_NUMBERS; zeta_index++) {

        let min_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) - 0.0) / f32(TOTAL_ZETA_NUMBERS)));
        let max_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) + 1.0) / f32(TOTAL_ZETA_NUMBERS)));
        let mid_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) + 0.5) / f32(TOTAL_ZETA_NUMBERS)));
        let period: f32 = f32(PERIODICITY) * pow(2,mid_zeta);

        for (var theta_index: u32 = 0u; theta_index < DIRECTIONS; theta_index++) {
            let theta = index_to_radians(theta_index);
            let dir = vec3<f32>(cos(theta), 0.0, sin(theta));

            // Sample amplitude at this position and direction
            let amplitude = sample_amplitude(real_x, real_z, theta_index, zeta_index);
            amplitude_sum += amplitude;

            // Position offset along wave direction
            let position_offset = dot(vec2<f32>(real_x, real_z), vec2<f32>(dir.x, dir.z)) * 2.0;

            // Profile buffer index (wrapping)
//            let profile_coord = ((wave_number * position_offset / TAU) + (DIRECTIONAL_OFFSETS[theta_index]/4.0)) * f32(PROFILE_BUFFER_LENGTH);
            let profile_coord = position_offset + (DIRECTIONAL_OFFSETS[theta_index]/1.0);
//            let profile_coord = position_offset;
//            let wrapped_index = i32((f32(PROFILE_BUFFER_LENGTH) * profile_coord / period)%f32(PROFILE_BUFFER_LENGTH));
//            let wrapped_index = i32(modulo((profile_coord / period) * f32(PROFILE_BUFFER_LENGTH), f32(PROFILE_BUFFER_LENGTH)));
            let index = i32(
                modulo(f32(PROFILE_BUFFER_LENGTH) * (f32(profile_coord) / f32(period)), f32(PROFILE_BUFFER_LENGTH))
            );
//            let sin_index =     i32(modulo((profile_coord / period)+ (period*(3.0/4.0)), f32(PROFILE_BUFFER_LENGTH)));

//            let profile_index = i32(profile_coord) % i32(PROFILE_BUFFER_LENGTH);
//            let wrapped_index = select(profile_index, profile_index + i32(PROFILE_BUFFER_LENGTH), profile_index < 0);

            // Index for 90-degree phase shift (sin from cos)
//            let sin_index = (wrapped_index + i32((3u * PROFILE_BUFFER_LENGTH) / 4u)) % i32(PROFILE_BUFFER_LENGTH);
//            let sin_index = (wrapped_index + i32((3.0 * period) / 4.0)) % i32(PROFILE_BUFFER_LENGTH);

            // Load profile values (cos and sin components)
//            let cos_val = textureLoad(profile_in, vec2<i32>(wrapped_index, i32(zeta_index))) * direction_multiplier;
            let cos_val = textureLoad(profile_in, vec2<i32>(index, i32(zeta_index))) * direction_multiplier;
//            let sin_val = textureLoad(profile_in, vec2<i32>(sin_index, i32(zeta_index))) * direction_multiplier;

            // Height contribution: Re(A) * cos - Im(A) * sin
            // This properly handles the complex amplitude with phase


            surface_offset.y += (amplitude * cos_val.y);

            surface_offset.x += dir.x*((amplitude * cos_val.x));
            surface_offset.z += dir.z*((amplitude * cos_val.x));

//            surface_offset.y += (amplitude.x * cos_val.y) - (amplitude.y * -cos_val.x);
//
//            surface_offset.x += dir.x*((amplitude.x * cos_val.x) - (amplitude.y * cos_val.y));
//            surface_offset.z += dir.z*((amplitude.x * cos_val.x) - (amplitude.y * cos_val.y));

//            surface_offset.y += (amplitude.x * cos_val.y);
//
//            surface_offset.x += dir.x*((amplitude.x * cos_val.x));
//            surface_offset.z += dir.z*((amplitude.x * cos_val.x));

//            tx.x += dir.x*((amplitude.x * cos_val.z) - (amplitude.y * sin_val.z));
//            tx.y += dir.x*((amplitude.x * cos_val.w) - (amplitude.y * sin_val.w));
//
//            tz.z += dir.z*((amplitude.x * cos_val.z) - (amplitude.y * sin_val.z));
//            tz.y += dir.z*((amplitude.x * cos_val.w) - (amplitude.y * sin_val.w));
        }
    }

    // Scale and store
    let normal = normalize(cross(tx, tz));

    let coord = vec2<i32>(i32(render_x), i32(render_z));
    textureStore(height_out, coord, vec4<f32>(surface_offset, 1.0));
//    textureStore(height_out, coord, vec4<f32>(vec3<f32>(0.0,surface_offset.y,0.0), 1.0));
//    textureStore(height_out, coord, vec4<f32>(vec3<f32>(0.0,amplitude_sum.x,0.0), 1.0));
    textureStore(normal_out, coord, vec4<f32>(normal, 1.0));
}
