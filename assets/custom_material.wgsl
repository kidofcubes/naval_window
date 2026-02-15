//#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::{mesh_functions::{get_world_from_local, mesh_position_local_to_clip, mesh_position_local_to_world}}
// we can import items from shader modules in the assets folder with a quoted path
//#import "shaders/custom_material_import.wgsl"::COLOR_MULTIPLIER
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
};

//struct FragmentInput {
//    @location(0) blend_color: vec4<f32>,
//}


//@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material_color: vec4<f32>;
//@group(#{MATERIAL_BIND_GROUP}) @binding(1) var material_color_texture: texture_2d<f32>;
//@group(#{MATERIAL_BIND_GROUP}) @binding(2) var material_color_sampler: sampler;

const PI = 3.1415926538;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

//    // Transform local position to clip space (screen space)
    out.clip_position = mesh_position_local_to_clip(
        get_world_from_local(vertex.instance_index),
//        vec4<f32>(vertex.position + vec3<f32>(0.0,sin(vertex.uv.x*5), 0.0), 1.0)
        vec4<f32>(vertex.position + vec3<f32>(0.0,2+sin(vertex.uv.x*2*PI)+sin(vertex.uv.y*2*PI),0.0), 1.0)
    );
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
//    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
//    sin(input.uv.x)
//    return vec4<f32>(sin(input.uv.x), sin(input.uv.y), 0.0, 1.0);
    return vec4<f32>(input.uv.xy, 0.0, 1.0);
}


//
//// For 2d replace `bevy_pbr::mesh_functions` with `bevy_sprite::mesh2d_functions`
//// and `mesh_position_local_to_clip` with `mesh2d_position_local_to_clip`.
//#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
//
//struct CustomMaterial {
//    color: vec4<f32>,
//};
//@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: CustomMaterial;
//
//struct Vertex {
//    @builtin(instance_index) instance_index: u32,
//    @location(0) position: vec3<f32>,
//    @location(1) blend_color: vec4<f32>,
//};
//
//struct VertexOutput {
//    @builtin(position) clip_position: vec4<f32>,
//    @location(0) blend_color: vec4<f32>,
//};
//
//@vertex
//fn vertex(vertex: Vertex) -> VertexOutput {
//    var out: VertexOutput;
//    out.clip_position = mesh_position_local_to_clip(
//        get_world_from_local(vertex.instance_index),
//        vec4<f32>(vertex.position, 1.0),
//    );
//    out.blend_color = vertex.blend_color;
//    return out;
//}
//
//struct FragmentInput {
//    @location(0) blend_color: vec4<f32>,
//};
//
//@fragment
//fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
//    return material.color * input.blend_color;
//}
