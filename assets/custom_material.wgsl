//#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::{mesh_functions::{get_world_from_local, mesh_position_local_to_clip, mesh_position_local_to_world}}
// we can import items from shader modules in the assets folder with a quoted path
//#import "shaders/custom_material_import.wgsl"::COLOR_MULTIPLIER

//for time since startup
//#import bevy_pbr::{mesh_view_bindings::globals}
//// this is made available to the importing module
const COLOR_MULTIPLIER: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 0.5);

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_position: vec4<f32>,
};

//struct FragmentInput {
//    @location(0) blend_color: vec4<f32>,
//}


//@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material_color: vec4<f32>;
//@group(#{MATERIAL_BIND_GROUP}) @binding(1) var material_color_texture: texture_2d<f32>;
//@group(#{MATERIAL_BIND_GROUP}) @binding(2) var material_color_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> time: f32;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> voxel_transform: mat4x4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> voxel_size: f32;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var voxel_texture: texture_3d<u32>;

@group(#{MATERIAL_BIND_GROUP}) @binding(5) var height_field: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var height_field_sampler: sampler;

@group(#{MATERIAL_BIND_GROUP}) @binding(7) var normal_field: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var normal_field_sampler: sampler;


const PI = 3.1415926538;
const TAU = PI * 2;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

//    // Transform local position to clip space (screen space)
    var timeMult = time * 1.0;
    var y_offset = 100.0;
    var offset: vec3<f32> = vec3<f32>(
        0.0,
//        100.0+sin((timeMult+vertex.uv.x)*TAU)+cos((timeMult+vertex.uv.z)*TAU),
        0.0+(0.0*0.5*(sin((timeMult+(vertex.uv.x*10.0))*TAU)+cos((timeMult+(vertex.uv.y*10.0))*TAU))),
//        1.3+(0.0*(sin((timeMult+(vertex.uv.x*10.0))*TAU)+cos((timeMult+(vertex.uv.y*10.0))*TAU))),
//        100.0+timeMult,
        0.0
    );
//    offset.y += textureSampleLevel(wake_texture, wake_sampler, vertex.uv.xy, 0.0).r * 10.0;
//    offset.y += textureSampleLevel(height_field, height_field_sampler, vertex.uv.xy, 0.0).x;
    offset += textureSampleLevel(height_field, height_field_sampler, vertex.uv.xy, 0.0).xyz;
//    offset.y += textureLoad(wake_texture, vec2<i32>(vertex.uv.xy*256), 0).r * 10;
    let world_position = vec4<f32>(vertex.position + offset,1.0) + vec4<f32>(0.0,y_offset,0.0,0.0);

    out.clip_position = mesh_position_local_to_clip(
        get_world_from_local(vertex.instance_index),
//        vec4<f32>(vertex.position + vec3<f32>(0.0,sin(vertex.uv.x*5), 0.0), 1.0)
        world_position
    );
//    out.world_position = world_position + vec4<f32>(0.0,10.0,0.0,0.0);
//    out.world_position = vec4<f32>(vertex.position, 1.0)+ vec4<f32>(0.0,3.0,0.0,0.0);
//    out.world_position = get_world_from_local(vertex.instance_index);
    let world_from_local = get_world_from_local(vertex.instance_index);
//    out.world_position = mesh_position_local_to_world(, vec4(vertex.position, 1.0));
    out.world_position = world_from_local * world_position;
    out.uv = vertex.uv;
//    out.blend_color = vec4<f32>(0.5,0.5,0.5, 0.5);

    // This is how bevy computes the world position
    // The vertex.instance_index is very important. Especially if you are using batching and gpu preprocessing
//    var world_from_local = get_world_from_local(vertex.instance_index);
//    var world_position = mesh_position_local_to_world(world_from_local, vec4(vertex.position, 1.0));
//    out.clip_position = position_world_to_clip(world_position.xyz);
//    out.uv = vertex.uv;

//    out.color = material.color;
    return out;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
//    return material_color * textureSample(material_color_texture, material_color_sampler, mesh.uv) * COLOR_MULTIPLIER;
    var hull_distance = 1.0;
    hull_distance = get_ship_distance_approx(input.world_position.xyz);
    hull_distance = clamp(hull_distance, 0.0, 3.0)/3.0;
    var color = vec4<f32>(26/255.0, 100/255.0, 200/255.0, 1.0);
//    var color = vec4<f32>(0.0,0.0,0.0, 1.0);
    color = color + (vec4<f32>(1.0) * (1.0-hull_distance));

//    color = color + vec4<f32>(input.uv.xy, input.world_position.y/10.0, 1.0);
    color = color + vec4<f32>(vec3<f32>(input.world_position.y/5.0), 1.0);
//    color = color + vec4<f32>(input.uv.xy-0.5, 0.0, 1.0);

    return color;
//    return vec4<f32>(input.world_position.xz, f32(center), 1.0);
//    return vec4<f32>(input.uv.xy, f32(), 1.0);
}


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