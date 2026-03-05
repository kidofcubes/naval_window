// Semi-Lagrangian advection of wave amplitudes
// Dispatched with workgroups (ceil(GRID_X/8), ceil(GRID_Z/8), DIRECTIONS)
// Each invocation handles one (x, z) cell for one direction

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
@group(0) @binding(4) var voxel_texture: texture_3d<u32>;
@group(0) @binding(5) var<uniform> voxel_settings: VoxelSettings;
@group(0) @binding(6) var<uniform> water_wave_group_speeds: WaterWaveGroupSpeeds;

// Hardware bilinear sampling using sampler
fn sample_amplitude(x: f32, z: f32, theta_index: u32, wave_index: u32) -> vec2<f32> {
    // Boundary check - return zero outside valid range
    if (x < 1.0 || x >= f32(GRID_X_SIZE - 1u) || z < 1.0 || z >= f32(GRID_Z_SIZE - 1u)) {
        return vec2<f32>(0.0, 0.0);
    }

    // Convert to normalized UV coordinates (0-1)
    // Add 0.5 to sample at texel centers
    let uv = vec2<f32>((f32(x) + 0.5) / f32(GRID_X_SIZE), (z + 0.5) / f32(GRID_Z_SIZE));
    let layer = layer_index(theta_index, wave_index);

    return textureSampleLevel(amplitude_in, amplitude_sampler, uv, layer, 0.0).xy;
}

// Transform water grid coordinates to world space
fn grid_to_voxel(grid_x: f32, grid_z: f32) -> vec3<f32> {
    let grid_pos = vec4<f32>(grid_x, -1.11, grid_z, 1.0);
    return (voxel_settings.water_to_voxel * grid_pos).xyz;
//    return (grid_pos).xyz;
}

// Check if a world position is inside a voxel boundary
fn is_boundary(voxel_pos: vec3<f32>) -> bool {
    let dims = vec3<i32>(voxel_settings.grid_dims);
    let voxel = vec3<i32>(voxel_pos);

    if (voxel.x < 0 || voxel.x >= dims.x ||
        voxel.y < 0 || voxel.y >= dims.y ||
        voxel.z < 0 || voxel.z >= dims.z) {
        return false;
    }

    let val = textureLoad(voxel_texture, voxel, 0).x;
    return (val & 2u) != 0u;  // VOXEL_BOUNDARY = 0b10
}

// Reflect direction by 180 degrees
fn reflect_direction(theta_index: u32) -> u32 {
    return (theta_index + DIRECTIONS / 2u) % DIRECTIONS;
}

@compute @workgroup_size(32, 32, 1)
fn advect(@builtin(global_invocation_id) global_id: vec3<u32>) {
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

        let group_speed = (water_wave_group_speeds.wave_group_speeds[zeta_index].x);

        // Displacement based on group velocity
        let dx = cos(theta) * group_speed * settings.delta_time;
        let dz = sin(theta) * group_speed * settings.delta_time;

        // Previous position (semi-Lagrangian)
        let prev_x = f32(x) - dx;
        let prev_z = f32(z) - dz;

        // Check if previous position is at a boundary
//        let world_pos = ;
        var new_amplitude: vec2<f32>;

//        if (is_boundary(grid_to_voxel(f32(x), f32(z)))) {
//            new_amplitude = vec2<f32>(0.0,0.0);
//        } else
//        if (is_boundary(grid_to_voxel(prev_x, prev_z))) {
        if (false) {
            // Reflect wave: sample from reflected direction with damping
            let reflected = reflect_direction(theta_index);
            let theta_r = index_to_radians(reflected);

            let drx = cos(theta_r) * group_speed * settings.delta_time;
            let drz = sin(theta_r) * group_speed * settings.delta_time;

            let prev_r_x = f32(x) - drx;
            let prev_r_z = f32(z) - drz;

            let reflected_amp = sample_amplitude(prev_r_x, prev_r_z, reflected, zeta_index);
            new_amplitude = reflected_amp * 1.0;
//            if(!is_boundary(grid_to_voxel(prev_r_x,prev_r_z))){
//                let reflected_amp = sample_amplitude(prev_r_x, prev_r_z, reflected, zeta_index);
//                new_amplitude = reflected_amp * 0.9;
//    //            let prev = sample_amplitude(prev_x, prev_z, theta_index, zeta_index);
//    //            new_amplitude = prev * 20.0;
//    //            new_amplitude = vec2<f32>(0.5,0.0);
//            }else{
//                new_amplitude = vec2<f32>(0.0,0.0);
//            }

//            new_amplitude = vec2<f32>(0.0,0.0);

        } else {
            // Normal advection - sample previous amplitude at current direction
            // Directional diffusion is now handled in a separate pass
            new_amplitude = sample_amplitude(prev_x, prev_z, theta_index, zeta_index);
        }

        // Write to output
        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer, vec4<f32>(new_amplitude, 0.0, 1.0));
    }
}
