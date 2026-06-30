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
    PERIODICITY, DIRECTIONAL_OFFSETS,
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

// FFT-based far-field wave field. Tiled at multiple physical scales (cascades).
const FFT_CASCADE_COUNT: u32 = 4u;
@group(#{MATERIAL_BIND_GROUP}) @binding(112) var fft_displacement: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(113) var fft_slope:        texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(114) var fft_sampler:      sampler;
// xyzw per cascade = (patch_length, k_min, k_max, _pad)
@group(#{MATERIAL_BIND_GROUP}) @binding(115) var<uniform> fft_cascade_params: array<vec4<f32>, FFT_CASCADE_COUNT>;
@group(#{MATERIAL_BIND_GROUP}) @binding(116) var<uniform> settings: WaterSettings;


fn modulo_f32(x: f32, n: f32) -> f32{
    return x - floor(x / n) * n;
}

// Hardware bilinear sample from amplitude texture array.
// Returns (real, imag) — GPU storage is r32float so .y is currently always 0,
// but we read both so the formula matches the CPU port verbatim.
fn sample_amplitude(x: f32, z: f32, theta_index: u32, zeta_index: u32) -> vec2<f32> {
    let uv = vec2<f32>((x + 0.5) / f32(GRID_X_SIZE), (z + 0.5) / f32(GRID_Z_SIZE));
    let layer = layer_index(theta_index, zeta_index);
    return textureSampleLevel(amplitude_texture, amplitude_sampler, uv, layer, 0.0).xy;
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

// Combined surface reconstruction: returns the world-space displacement of
// this UV column AND the surface normal in one pass, mirroring the CPU
// `reconstruct_height` function in cpu_debug_impl.rs.
struct WaterSurface {
    offset: vec3<f32>,
    normal: vec3<f32>,
}

// Sum the displacement and slope of every FFT cascade at this world XZ.
// Cascades tile at different physical scales so no global repeat is visible.
fn sample_fft_surface(world_xz: vec2<f32>) -> WaterSurface {
    var total_offset = vec3<f32>(0.0, 0.0, 0.0);
    var total_slope  = vec2<f32>(0.0, 0.0);

    for (var c: u32 = 0u; c < FFT_CASCADE_COUNT; c = c + 1u) {
        let patch_unreserved_keyword = fft_cascade_params[c].x;
        let uv = fract(world_xz / patch_unreserved_keyword + vec2<f32>(0.5));
        let disp  = textureSampleLevel(fft_displacement, fft_sampler, uv, c, 0.0);
        let slope = textureSampleLevel(fft_slope,        fft_sampler, uv, c, 0.0);
        total_offset += vec3<f32>(disp.x, disp.y, disp.z);
        total_slope  += vec2<f32>(slope.x, slope.y);
    }

    var result: WaterSurface;
    result.offset = total_offset;
    result.normal = normalize(vec3<f32>(-total_slope.x, 1.0, -total_slope.y));
    return result;
}

fn calculate_water_surface(uv: vec2<f32>) -> WaterSurface {
    var result: WaterSurface;
    result.offset = vec3<f32>(0.0, 0.0, 0.0);
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || true){
        // Outside the realwaves grid → fall back to the tiled FFT cascades.
        let world_xz = vec2<f32>(
            (uv.x - 0.5) * f32(GRID_X_SIZE) * f32(GRID_SCALE),
            (uv.y - 0.5) * f32(GRID_Z_SIZE) * f32(GRID_SCALE),
        );
        return sample_fft_surface(world_xz);
    }

    // Map UV coordinates to grid coordinates
    let real_x = uv.x * f32(GRID_X_SIZE);
    let real_z = uv.y * f32(GRID_Z_SIZE);

    var surface_offset = vec3<f32>(0.0, 0.0, 0.0);
    var tx = vec3<f32>(0.0, 0.0, 0.0);
    var tz = vec3<f32>(0.0, 0.0, 0.0);

    let direction_multiplier = 1.0 / f32(DIRECTIONS);
    let zeta_full_range = settings.zeta_max - settings.zeta_min;

    for (var zeta_index: u32 = 0u; zeta_index < TOTAL_ZETA_NUMBERS; zeta_index++) {
        let mid_zeta = settings.zeta_min + (zeta_full_range * ((f32(zeta_index) + 0.5) / f32(TOTAL_ZETA_NUMBERS)));
        let period: f32 = f32(PERIODICITY) * pow(2.0, mid_zeta);

        for (var theta_index: u32 = 0u; theta_index < DIRECTIONS; theta_index++) {
            let theta = index_to_radians(theta_index);
            let dir = vec3<f32>(cos(theta), 0.0, sin(theta));

            // amplitude = (real, imag) — see sample_amplitude
            let amplitude = sample_amplitude(real_x, real_z, theta_index, zeta_index);

            // Profile lookup, scaled by 1/DIRECTIONS so per-direction contributions sum to 1.
            let cos_val = sample_profile(real_x, real_z, theta_index, zeta_index, period, vec2<f32>(dir.x, dir.z)) * direction_multiplier;

            // ── displacement ────────────────────────────────────────────────
            // Height contribution: Re(A) * cos - Im(A) * (-sin)
            surface_offset.y += (amplitude.x * cos_val.y) - (amplitude.y * -cos_val.x);

            let horiz_disp = (amplitude.x * cos_val.x) - (amplitude.y * cos_val.y);
            surface_offset.x += dir.x * horiz_disp;
            surface_offset.z += dir.z * horiz_disp;

            // ── tangent derivatives w.r.t. wave parameter p ─────────────────
            let dh_dp = (amplitude.x * cos_val.z) - (amplitude.y * cos_val.w);
            let dv_dp = (amplitude.x * cos_val.w) - (amplitude.y * cos_val.z);

            // d(world)/dx with dp/dx = dir.x (identity +X added after the loop)
            tx += vec3<f32>(dir.x * dir.x * dh_dp, dir.x * dv_dp, dir.x * dir.z * dh_dp);
            // d(world)/dz with dp/dz = dir.z (identity +Z added after the loop)
            tz += vec3<f32>(dir.x * dir.z * dh_dp, dir.z * dv_dp, dir.z * dir.z * dh_dp);
        }
    }

    // Add the grid identity term (undisplaced surface tangents).
    tx += vec3<f32>(1.0, 0.0, 0.0);
    tz += vec3<f32>(0.0, 0.0, 1.0);

    result.offset = surface_offset * GRID_SCALE;
    result.normal = normalize(cross(tz, tx));
    return result;
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
    var offset: vec3<f32> = calculate_water_surface(uv_from_world(orig_world_position.xyz)).offset * 1.0;
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
    var world_normal = calculate_water_surface(uv_from_world(in.world_position.xyz)).normal;
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