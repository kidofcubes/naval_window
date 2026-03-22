// Semi-Lagrangian advection of wave amplitudes
// Dispatched with workgroups (ceil(GRID_X/8), ceil(GRID_Z/8), DIRECTIONS)
// Each invocation handles one (x, z) cell for one direction

#import "water_common.wgsl"::{
    GRID_X_SIZE, GRID_Z_SIZE, GRID_SCALE, RENDER_X_SIZE, RENDER_Z_SIZE,
    DIRECTIONS, TOTAL_ZETA_NUMBERS, PROFILE_BUFFER_LENGTH,
    ZETA_MIN, ZETA_MAX, DIRECTIONAL_OFFSETS,
    GRAVITY, TIME_MULT, TAU, WaterSettings, WaterDebugSettings, VoxelSettings, WaterWaveGroupSpeeds,
    index_to_radians, layer_index
}

@group(0) @binding(0) var amplitude_in: texture_2d_array<f32>;
@group(0) @binding(1) var amplitude_sampler: sampler;
@group(0) @binding(2) var amplitude_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(3) var<uniform> settings: WaterSettings;
@group(0) @binding(4) var voxel_texture: texture_3d<u32>;
@group(0) @binding(5) var<uniform> voxel_settings: VoxelSettings;
@group(0) @binding(6) var<uniform> water_wave_group_speeds: WaterWaveGroupSpeeds;
@group(0) @binding(7) var sdf_texture: texture_3d<f32>;

// Hardware bilinear sampling using sampler
fn sample_amplitude(x: f32, z: f32, theta_index: u32, wave_index: u32) -> f32 {
    // Boundary check - return zero outside valid range
    if (x < 1.0 || x >= f32(GRID_X_SIZE - 1u) || z < 1.0 || z >= f32(GRID_Z_SIZE - 1u)) {
//        return vec2<f32>(0.0, 0.0);
        return 0.0;
    }

    // Convert to normalized UV coordinates (0-1)
    // Add 0.5 to sample at texel centers
    let uv = vec2<f32>((f32(x) + 0.5) / f32(GRID_X_SIZE), (z + 0.5) / f32(GRID_Z_SIZE));
    let layer = layer_index(theta_index, wave_index);

    return textureSampleLevel(amplitude_in, amplitude_sampler, uv, layer, 0.0).x;
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

// Sample the SDF at a voxel position
fn sample_sdf(voxel_pos: vec3<f32>) -> f32 {
    let dims = vec3<i32>(voxel_settings.grid_dims);
    let voxel = vec3<i32>(voxel_pos);
    if (voxel.x < 0 || voxel.x >= dims.x ||
        voxel.y < 0 || voxel.y >= dims.y ||
        voxel.z < 0 || voxel.z >= dims.z) {
        return 1000.0; // large positive = far outside
    }
    return textureLoad(sdf_texture, voxel, 0).x;
}

// Compute the SDF gradient (surface normal) via central differences
fn sdf_normal(voxel_pos: vec3<f32>) -> vec3<f32> {
    let eps = 1.0; // 1 voxel
    let dx = sample_sdf(voxel_pos + vec3<f32>(eps, 0.0, 0.0)) - sample_sdf(voxel_pos - vec3<f32>(eps, 0.0, 0.0));
    let dy = sample_sdf(voxel_pos + vec3<f32>(0.0, eps, 0.0)) - sample_sdf(voxel_pos - vec3<f32>(0.0, eps, 0.0));
    let dz = sample_sdf(voxel_pos + vec3<f32>(0.0, 0.0, eps)) - sample_sdf(voxel_pos - vec3<f32>(0.0, 0.0, eps));
    let grad = vec3<f32>(dx, dy, dz);
    let len = length(grad);
    if (len < 0.0001) {
        return vec3<f32>(0.0, 1.0, 0.0); // fallback
    }
    return grad / len;
}

// Find the closest theta_index for a given 2D direction on the XZ plane
fn direction_to_theta_index(dir: vec2<f32>) -> u32 {
    let angle = atan2(dir.y, dir.x); // range [-pi, pi]
    // Map to [0, TAU)
    var a = angle;
    if (a < 0.0) {
        a += TAU;
    }
    let index = u32(round(a / TAU * f32(DIRECTIONS))) % DIRECTIONS;
    return index;
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

        // SDF-based reflection check
        let voxel_pos = grid_to_voxel(f32(x), f32(z));
        let sdf_val = sample_sdf(voxel_pos);
        var new_amplitude: f32;

        if (sdf_val < 2.0) {
            // Near or inside boundary — reflect using SDF gradient
            let normal_3d = sdf_normal(voxel_pos);
            // Project to XZ plane for 2D wave reflection
            let n_xz = normalize(vec2<f32>(normal_3d.x, normal_3d.z) + vec2<f32>(0.0001, 0.0));

            // Current wave direction in 2D
            let d = vec2<f32>(cos(theta), sin(theta));

            // Reflect: d_reflected = d - 2*dot(d, n)*n
            let d_reflected = d - 2.0 * dot(d, n_xz) * n_xz;

            // Find closest discrete theta index for reflected direction
            let reflected_index = direction_to_theta_index(d_reflected);
            let theta_r = index_to_radians(reflected_index);

            // Trace back from reflected direction
            let drx = cos(theta_r) * group_speed * settings.delta_time;
            let drz = sin(theta_r) * group_speed * settings.delta_time;
            let prev_r_x = f32(x) - drx;
            let prev_r_z = f32(z) - drz;

            let reflected_amp = sample_amplitude(prev_r_x, prev_r_z, reflected_index, zeta_index);
            new_amplitude = reflected_amp * 0.9; // damping factor
        } else {
            // Normal advection - sample previous amplitude at current direction
            new_amplitude = sample_amplitude(prev_x, prev_z, theta_index, zeta_index);
        }
        if (is_boundary(grid_to_voxel(f32(x), f32(z)))) {
            new_amplitude = 1000.0;
        }

        // Write to output
        textureStore(amplitude_out, vec2<i32>(i32(x), i32(z)), layer, vec4<f32>(new_amplitude, 0.0, 0.0, 1.0));
    }
}
