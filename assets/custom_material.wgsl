//#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::{mesh_functions::{get_world_from_local, mesh_position_local_to_clip, mesh_position_local_to_world}}

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    view_transformations::position_world_to_clip,
}


#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif


// we can import items from shader modules in the assets folder with a quoted path
//#import "shaders/custom_material_import.wgsl"::COLOR_MULTIPLIER

#import "water_common.wgsl"::{
    GRID_X_SIZE, GRID_Z_SIZE, GRID_SCALE, RENDER_X_SIZE, RENDER_Z_SIZE,
    DIRECTIONS, TOTAL_ZETA_NUMBERS, PROFILE_BUFFER_LENGTH,
    PERIODICITY, ZETA_MIN, ZETA_MAX, DIRECTIONAL_OFFSETS,
    GRAVITY, TIME_MULT, TAU, WaterSettings, WaterDebugSettings, VoxelSettings, WaterWaveGroupSpeeds,
    index_to_radians, layer_index
}

//for time since startup
//#import bevy_pbr::{mesh_view_bindings::globals}
//// this is made available to the importing module
const COLOR_MULTIPLIER: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 0.5);

//struct Vertex {
//    @builtin(instance_index) instance_index: u32,
//    @location(0) position: vec3<f32>,
//    @location(1) uv: vec2<f32>,
//};
//
//struct VertexOutput {
//    @builtin(position) clip_position: vec4<f32>,
//    @location(0) uv: vec2<f32>,
//    @location(1) world_position: vec4<f32>,
//};

//struct FragmentInput {
//    @location(0) blend_color: vec4<f32>,
//}


//@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material_color: vec4<f32>;
//@group(#{MATERIAL_BIND_GROUP}) @binding(1) var material_color_texture: texture_2d<f32>;
//@group(#{MATERIAL_BIND_GROUP}) @binding(2) var material_color_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> time: f32;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<uniform> voxel_transform: mat4x4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> voxel_size: f32;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var voxel_texture: texture_3d<u32>;

@group(#{MATERIAL_BIND_GROUP}) @binding(109 ) var amplitude_texture: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(110) var amplitude_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(111) var profile_buffer:     texture_2d<f32>;



// Hardware bilinear sample from amplitude texture array
fn sample_amplitude(x: f32, z: f32, theta_index: u32, zeta_index: u32) -> f32 {
    // Convert to normalized UV coordinates (0-1)
    // Add 0.5 to sample at texel centers
    let uv = vec2<f32>((x + 0.5) / f32(GRID_X_SIZE), (z + 0.5) / f32(GRID_Z_SIZE));
    let layer = layer_index(theta_index, zeta_index);

    return textureSampleLevel(amplitude_texture, amplitude_sampler, uv, layer, 0.0).x;
}

fn modulo_f32(x: f32, n: f32) -> f32{
    return x - floor(x / n) * n;
}

// Calculate water surface offset at a given world position
fn calculate_water_offset(uv: vec2<f32>) -> vec3<f32> {
    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0){
        return vec3<f32>(0.0,0.0,0.0);
    }
    // Map UV coordinates to grid coordinates
    let real_x = uv.x * f32(GRID_X_SIZE);
    let real_z = uv.y * f32(GRID_Z_SIZE);

    var surface_offset = vec3<f32>(0.0, 0.0, 0.0);
    var tx = vec3<f32>(0.0, 0.0, 0.0);
    var tz = vec3<f32>(0.0, 0.0, 0.0);

    let direction_multiplier = 1.0/f32(DIRECTIONS);
    let zeta_full_range = ZETA_MAX - ZETA_MIN;

    for (var zeta_index: u32 = 0u; zeta_index < TOTAL_ZETA_NUMBERS; zeta_index++) {
        let min_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) - 0.0) / f32(TOTAL_ZETA_NUMBERS)));
        let max_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) + 1.0) / f32(TOTAL_ZETA_NUMBERS)));
        let period: f32 = f32(PERIODICITY) * pow(2,max_zeta);

        for (var theta_index: u32 = 0u; theta_index < DIRECTIONS; theta_index++) {
            let theta = index_to_radians(theta_index);
            let dir = vec3<f32>(cos(theta), 0.0, sin(theta));

            // Sample amplitude at this position and direction
            let amplitude = sample_amplitude(real_x, real_z, theta_index, zeta_index);

            // Position offset along wave direction
            let position_offset = dot(vec2<f32>(real_x, real_z), vec2<f32>(dir.x, dir.z)) * GRID_SCALE;

            // Profile buffer index (wrapping)
            let profile_coord = position_offset + (DIRECTIONAL_OFFSETS[theta_index]/1.0);
            let index = i32(
                modulo_f32(f32(PROFILE_BUFFER_LENGTH) * (f32(profile_coord) / f32(period)), f32(PROFILE_BUFFER_LENGTH))
            );

            // Load profile value (cos component)
            let cos_val = textureLoad(profile_buffer, vec2<i32>(index, i32(zeta_index)), 0) * direction_multiplier;

            // Height contribution
            surface_offset.y += (amplitude * cos_val.y);
            surface_offset.x += dir.x * (amplitude * cos_val.x);
            surface_offset.z += dir.z * (amplitude * cos_val.x);
        }
    }
//    return vec3<f32>(0.0,sample_amplitude(real_x,real_z,0,0),0.0);

    return surface_offset / GRID_SCALE;
}

//// Calculate water normal at a given world position using finite differences
//fn calculate_water_normal(world_pos: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
//    let epsilon = 0.1; // Small offset for finite difference
//
//    // Sample height at neighboring points
//    let center = calculate_water_offset(world_pos.x, world_pos.z, uv);
//    let right = calculate_water_offset(uv + vec2(epsilon/400.0, 0.0));
//    let up = calculate_water_offset(world_pos.x, world_pos.z + epsilon, uv + vec2(0.0, epsilon/400.0));
//
//    // Calculate tangent vectors
//    let tangent_x = vec3(epsilon, right.y - center.y, 0.0);
//    let tangent_z = vec3(0.0, up.y - center.y, epsilon);
//
//    // Cross product gives normal
//    let normal = normalize(cross(tangent_x, tangent_z));
//
//    return normal;
//}


//const PI = 3.1415926538;
//const TAU = PI * 2;

//@vertex
//fn prepass_vertex(vertex: Vertex) -> VertexOutput {
//    return translate_vertex(vertex);
//}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    return translate_vertex(vertex);
}
fn translate_vertex(vertex: Vertex) -> VertexOutput {
    let world_from_local = get_world_from_local(vertex.instance_index);
    let orig_world_position = world_from_local * vec4(vertex.position,1.0);
    var out: VertexOutput;

    // Transform local position to clip space (screen space)
    var timeMult = time * 1.0;
    var y_offset = 100.0;

    // Calculate water displacement using UV coordinates
    var offset: vec3<f32> = calculate_water_offset(
        vec2<f32>((orig_world_position.x/1024.0)+0.5, (orig_world_position.z/1024.0)+0.5)
//       vertex.uv
    ) * 1.0;

    let position = vec4<f32>(vertex.position + offset,1.0) + vec4<f32>(0.0,y_offset,0.0,0.0);

    out.position = mesh_position_local_to_clip(
        world_from_local,
        position
    );

    out.world_position = world_from_local * position;
    out.uv = vertex.uv;

    return out;
}

//@fragment
//fn fragment(
//    in: VertexOutput,
//    @builtin(front_facing) is_front: bool,
//) -> FragmentOutput {
//
//    // generate a PbrInput struct from the StandardMaterial bindings
//    var pbr_input = pbr_input_from_standard_material(in, is_front);
//#ifdef PREPASS_PIPELINE
//    pbr_input.world_normal = normalize(vec3(1.0,1.0,1.0));
//    let out = deferred_output(in, pbr_input);
//#else
//    var out: FragmentOutput;
//    pbr_input.world_normal = normalize(vec3(1.0,1.0,1.0));
//    out.color = apply_pbr_lighting(pbr_input);
//    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
//#endif
//    return out;
////
////
////    // Calculate per-pixel normal using finite differences
//////    let normal = calculate_water_normal(input.world_position.xyz, input.uv);
////
////    var hull_distance = 1.0;
////    hull_distance = get_ship_distance_approx(input.world_position.xyz);
////    hull_distance = clamp(hull_distance, 0.0, 3.0)/3.0;
////    var color = vec4<f32>(26/255.0, 100/255.0, 200/255.0, 1.0);
////    color = color + (vec4<f32>(1.0) * (1.0-hull_distance));
////    color = color + vec4<f32>(vec3<f32>(input.world_position.y/5.0), 1.0);
////
////    return color;
//}


fn get_ship_distance_approx(world_pos: vec3<f32>) -> f32 {
    let voxel_pos = vec3<i32>((voxel_transform * vec4<f32>(world_pos, 1.0)).xyz);
    let dims = vec3<i32>(textureDimensions(voxel_texture));

    if(voxel_pos.x < 0 || voxel_pos.x >= dims.x || voxel_pos.y < 0 || voxel_pos.y >= dims.y || voxel_pos.z < 0 || voxel_pos.z >= dims.z){
        return 999.0;
    }

    // Check if we're inside
    let center = textureLoad(voxel_texture, voxel_pos, 0).x;
    if (center > 0u) {
        return 0.0; // Inside ship
    }

//    // Check 6-neighbor distance
    var min_dist = 999.0;
//    for (var x: i32 = -1; x <= 1; x++) {
//        for (var y: i32 = -1; y <= 1; y++) {
//            for (var z: i32 = -1; z <= 1; z++) {
//                let neighbor_coords = voxel_pos + vec3<i32>(x, y, z);
//                let neighbor = textureLoad(voxel_texture, neighbor_coords, 0).r;
//                if (neighbor > 0u) {
//                    let dist = length(vec3<f32>(f32(x), f32(y), f32(z))) * voxel_size;
//                    min_dist = min(min_dist, dist);
//                }
//            }
//        }
//    }

    return min_dist;
}

fn inverse(m: mat4x4f) -> mat4x4f {
    let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
    let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
    let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
    let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    return mat4x4f(
        a11 * b11 - a12 * b10 + a13 * b09,
        a02 * b10 - a01 * b11 - a03 * b09,
        a31 * b05 - a32 * b04 + a33 * b03,
        a22 * b04 - a21 * b05 - a23 * b03,
        a12 * b08 - a10 * b11 - a13 * b07,
        a00 * b11 - a02 * b08 + a03 * b07,
        a32 * b02 - a30 * b05 - a33 * b01,
        a20 * b05 - a22 * b02 + a23 * b01,
        a10 * b10 - a11 * b08 + a13 * b06,
        a01 * b08 - a00 * b10 - a03 * b06,
        a30 * b04 - a31 * b02 + a33 * b00,
        a21 * b02 - a20 * b04 - a23 * b00,
        a11 * b07 - a10 * b09 - a12 * b06,
        a00 * b09 - a01 * b07 + a02 * b06,
        a31 * b01 - a30 * b03 - a32 * b00,
        a20 * b03 - a21 * b01 + a22 * b00) * (1 / det);
}