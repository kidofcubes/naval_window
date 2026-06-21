#import bevy_pbr::{mesh_functions::{get_world_from_local, mesh_position_local_to_clip, mesh_position_local_to_world}}

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}


#import bevy_pbr::{
    prepass_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}

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


fn modulo_f32(x: f32, n: f32) -> f32{
    return x - floor(x / n) * n;
}

// Hardware bilinear sample from amplitude texture array
fn sample_amplitude(x: f32, z: f32, theta_index: u32, zeta_index: u32) -> f32 {
    // Convert to normalized UV coordinates (0-1)
    // Add 0.5 to sample at texel centers

    //when x is 512.0 and z is 512.0, we should end up sampling from the middle of pixel 512,512

    let uv = vec2<f32>((x + 0.5) / f32(GRID_X_SIZE), (z + 0.5) / f32(GRID_Z_SIZE));
    let layer = layer_index(theta_index, zeta_index);

    //will be 0.0 from 511.5 to 512.5 in index coordinates (true 0.5 around the center)
//    if (i32(round(x))==512||i32(round(z))==512){
//        return 0.0;
//    }

//    if (i32(x)==511||i32(z)==511){
//        return 0.0;
//    }

    return textureSampleLevel(amplitude_texture, amplitude_sampler, uv, layer, 0.0).x;

    //when x is 512 and z is 512, should be
//    return textureLoad(amplitude_texture, vec2<i32>(i32(floor(x)),i32(floor(z))), layer, 0).x;
}

fn sample_profile(x: f32, z: f32, theta_index: u32, zeta_index: u32, period: f32, dir: vec2<f32>) -> vec4<f32>{
    let position_offset = dot(vec2<f32>(x, z), dir) * GRID_SCALE;

    // Profile buffer index (wrapping)
    let profile_coord = position_offset + (DIRECTIONAL_OFFSETS[theta_index]/1.0);
    let index = i32(
        modulo_f32(f32(PROFILE_BUFFER_LENGTH) * (f32(profile_coord) / f32(period)), f32(PROFILE_BUFFER_LENGTH))
    );

    // Load profile value
    return textureLoad(profile_buffer, vec2<i32>(index, i32(zeta_index)), 0);
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
//    var tx = vec3<f32>(0.0, 0.0, 0.0);
//    var tz = vec3<f32>(0.0, 0.0, 0.0);

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

            let profile = sample_profile(real_x, real_z, theta_index, zeta_index, period, vec2<f32>(dir.x,dir.z)) * direction_multiplier;

            // Height contribution
            surface_offset.y += (amplitude * profile.y);
            surface_offset.x += dir.x * (amplitude * profile.x);
            surface_offset.z += dir.z * (amplitude * profile.x);
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

fn compute_normal(uv: vec2<f32>) -> vec3<f32> {
//    return vec3<f32>(0.0,1.0,0.0);
    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0){
        return vec3<f32>(0.0,1.0,0.0);
    }
    // Map UV coordinates to grid coordinates
    let real_x = uv.x * f32(GRID_X_SIZE);
    let real_z = uv.y * f32(GRID_Z_SIZE);

//    var grad = vec2<f32>(0.0, 0.0);
    var tx = vec3<f32>(0.0, 0.0, 0.0);
    var tz = vec3<f32>(0.0, 0.0, 0.0);

    var dhdx = 0.0;
    var dhdz = 0.0;

    let zeta_full_range = ZETA_MAX - ZETA_MIN;
    for (var zeta_index: u32 = 0u; zeta_index < TOTAL_ZETA_NUMBERS; zeta_index++) {
        let min_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) - 0.0) / f32(TOTAL_ZETA_NUMBERS)));
        let max_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) + 1.0) / f32(TOTAL_ZETA_NUMBERS)));
        let period: f32 = f32(PERIODICITY) * pow(2,max_zeta);
        let k        = (2.0 * 3.14159265) / period;  // wavenumber for this band

        for (var theta_index: u32 = 0u; theta_index < DIRECTIONS; theta_index++) {
            let theta = index_to_radians(theta_index);
            let dir = vec3<f32>(cos(theta), 0.0, sin(theta));

            let amplitude = sample_amplitude(real_x,real_z, theta_index, zeta_index);

            let profile = sample_profile(real_x, real_z, theta_index, zeta_index, period, vec2<f32>(dir.x,dir.z));
            tx += dir.x * vec3<f32>(
                amplitude * profile.z,
                amplitude * profile.w,
                0.0
            );
            tz += dir.z * vec3<f32>(
                0.0,
                amplitude * profile.w,
                amplitude * profile.z
            );
            dhdx += amplitude * profile.w * k * dir.x;
            dhdz += amplitude * profile.w * k * dir.z;
        }
    }
    if((tx.x == 0.0 && tx.y == 0.0 && tx.z == 0.0)||(tz.x == 0.0 && tz.y == 0.0 && tz.z == 0.0)){
        return vec3<f32>(0.0,1.0,0.0);
    }
    var n = normalize(cross(tz,tx));
    return select(-n, n, n.y > 0.0);
//    return normalize(vec3<f32>(-dhdx, 1.0, -dhdz));
}


fn uv_from_world(world_pos: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(
        (world_pos.x / (f32(GRID_X_SIZE) * f32(GRID_SCALE)))+0.5,
        (world_pos.z / (f32(GRID_Z_SIZE) * f32(GRID_SCALE)))+0.5
    );
}

// f0 for water is ~0.02 — the reflectance at normal incidence.
// This is the physical value for an air/water interface.
const WATER_F0: f32 = 0.02;

// Deep ocean absorbs almost all light — this should be very close to black.
// The tiny blue tint is the small amount of blue light that scatters back
// before being absorbed.
const DEEP_COLOR: vec3<f32>    = vec3<f32>(0.002, 0.006, 0.012);

// At steep viewing angles you see slightly more scattering — still very dark,
// just a touch more blue than the deep color. Not teal, not bright.
const SHALLOW_COLOR: vec3<f32> = vec3<f32>(0.004, 0.012, 0.025);

// ── Fresnel ──────────────────────────────────────────────────────────────────

// Schlick approximation of the Fresnel equations.
// cos_theta is dot(view_direction, surface_normal), clamped to [0,1].
// Returns a value in [f0, 1.0] — higher at grazing angles.
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

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
    var offset: vec3<f32> = calculate_water_offset(uv_from_world(orig_world_position.xyz)) * 1.0;
//    offset = vec3<f32>(0.0,0.0,0.0);

    let position = vec4<f32>(vertex.position + offset,1.0) + vec4<f32>(0.0,y_offset,0.0,0.0);

    out.position = mesh_position_local_to_clip(
        world_from_local,
        position
    );

    out.world_position = world_from_local * position;
    out.uv = vertex.uv;
//    out.world_normal = vec3(0.0,0.0,0.0);

//#ifdef PREPASS_PIPELINE
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
//    out.world_normal = vertex.normal;
#endif

    return out;
}

//#ifdef DEFERRED_PREPASS
//#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
//#ifndef NORMAL_PREPASS_OR_DEFERRED_PREPASS
//#ifdef FRAGMENT
#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var out: FragmentOutput;
    var pbr_input = pbr_input_from_standard_material(in, is_front);
//    pbr_input.material.base_color.b = 255;
//#ifdef NORMAL_PREPASS
    var world_normal = compute_normal(uv_from_world(in.world_position.xyz));
//    pbr_input.material.base_color.r = world_normal.x;
//    pbr_input.material.base_color.g = world_normal.y;
//    pbr_input.material.base_color.b = world_normal.z;
//    world_normal = vec3<f32>(world_normal.z,world_normal.y,-world_normal.x);
//    world_normal.y *= -1.0;
//    world_normal = vec3<f32>(-1.0,-1.0,0.0);
//    pbr_input.world_normal = world_normal;
    pbr_input.N = normalize(
        (view.view_from_world * vec4<f32>(world_normal, 0.0)).xyz
    );
//    pbr_input.N = vec3<f32>(0.0,1.0,0.0);
    pbr_input.N = world_normal;
//#endif


    // ── Fresnel ───────────────────────────────────────────────────────────────

    // Reconstruct the view direction in world space.
    // view.world_position is the camera's world-space position.
    let view_dir_world = normalize(
        view.world_position.xyz - in.world_position.xyz
    );

    // cos_theta: how directly the camera is looking at this fragment's surface.
    // 1.0 = straight down, 0.0 = perfectly grazing.
    let cos_theta = saturate(dot(world_normal, view_dir_world));

    let fresnel = fresnel_schlick(cos_theta, WATER_F0);

    // ── Base color blend ──────────────────────────────────────────────────────

    // At grazing angles (fresnel → 1.0) the surface is highly reflective —
    // the environment map handles that, so the base color should go dark.
    // At steep angles (fresnel → 0.0) you see through into the water — use
    // the shallow/deep water color.
    //
    // We set base_color to a blend driven by (1 - fresnel) so that:
    //   - grazing angle → dark base (environment reflection dominates)
    //   - steep angle   → water interior color visible
    var water_color = mix(SHALLOW_COLOR, DEEP_COLOR, fresnel);

    var hull_distance = 1.0;
    hull_distance = get_ship_distance_approx(in.world_position.xyz);
    hull_distance = clamp(hull_distance, 0.0, 3.0)/3.0;
//    hull_distance = 0.0;
    water_color = mix(vec3<f32>(1.0), water_color, hull_distance);


    // Write the blended color as base_color. Bevy's PBR then mixes this
    // with the environment map reflection via the reflectance / roughness
    // parameters you already have set.
    pbr_input.material.base_color = vec4<f32>(water_color, 1.0);

    // Drive Bevy's own Fresnel/reflectance with the same factor so the
    // specular highlight intensity also scales with viewing angle.
    // perceptual_roughness and reflectance were set on the Rust material side
    // and are already in pbr_input.material — no need to override them here
    // unless you want to vary roughness across the surface (e.g. foam areas).

    // generate a PbrInput struct from the StandardMaterial bindings
//#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
//    pbr_input.world_normal = normalize(vec3(1.0,1.0,1.0));
    out = deferred_output(in, pbr_input);
//#else
//    pbr_input.world_normal = normalize(vec3(1.0,1.0,1.0));
//    out.color = apply_pbr_lighting(pbr_input);
//    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
//#endif
    return out;

//
//
//    // Calculate per-pixel normal using finite differences
////    let normal = calculate_water_normal(input.world_position.xyz, input.uv);

//
//    return color;
}
#endif


fn get_ship_distance_approx(world_pos: vec3<f32>) -> f32 {
    let voxel_pos = vec3<i32>((voxel_transform * vec4<f32>(world_pos, 1.0)).xyz);
//    var voxel_pos = vec3<i32>(world_pos);
//    voxel_pos.y += 3;
    let dims = vec3<i32>(textureDimensions(voxel_texture));

    if(voxel_pos.x < 0 || voxel_pos.x >= dims.x || voxel_pos.y < 0 || voxel_pos.y >= dims.y || voxel_pos.z < 0 || voxel_pos.z >= dims.z){
        return 999.0;
    }

    // Check if we're inside
    let center = textureLoad(voxel_texture, voxel_pos, 0).x;
    if ((center & 2u)>0) { //boundary voxel
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