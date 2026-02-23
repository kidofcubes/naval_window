// ─────────────────────────────────────────────────────────────────────────────
// water_surface.wgsl
//
// Vertex + Fragment shader for the water surface mesh.
//
// Vertex stage:
//   • Samples the height field texture at the mesh vertex's XZ world position.
//   • Displaces the vertex in Y.
//   • Computes analytical normals from finite differences of the height field.
//
// Fragment stage:
//   • PBR-style water shading:
//       – Fresnel-blended reflection/refraction approximation.
//       – Depth-based colour (shallow=turquoise, deep=navy).
//       – Specular highlight from sun direction.
//       – Foam mask where wave crests exceed a threshold.
// ─────────────────────────────────────────────────────────────────────────────

#import bevy_pbr::{mesh_functions::{get_world_from_local, mesh_position_local_to_clip, mesh_position_local_to_world, mesh_normal_local_to_world}}
#import bevy_pbr::view_transformations
#import bevy_pbr::position_world_to_clip
//#import bevy_render::view  View
#import bevy_render::view::View

// ─── Material Uniforms ────────────────────────────────────────────────────────

struct WaterMaterial {
    base_color:   vec4<f32>,
    roughness:    f32,
    height_scale: f32,
    domain_size:  f32,
    grid_size:    u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: WaterMaterial;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var height_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var height_sampler: sampler;

// ─── Vertex I/O ───────────────────────────────────────────────────────────────

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos:   vec4<f32>,
    @location(0)       world_pos:  vec3<f32>,
    @location(1)       uv:         vec2<f32>,
    @location(2)       wave_height: f32,
    @location(3)       normal:     vec3<f32>,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Sample height field at UV (0..1 over the whole domain).
fn sample_height(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(height_tex, height_sampler, uv, 0.0).r;
}

/// Compute water-surface normal via central finite differences on the height field.
fn compute_normal(uv: vec2<f32>) -> vec3<f32> {
    let cell_uv = 1.0 / f32(material.grid_size);  // one texel in UV space
    let hs      = material.height_scale;
    // Sample the four neighbours.
    let hL = sample_height(uv + vec2<f32>(-cell_uv, 0.0)) * hs;
    let hR = sample_height(uv + vec2<f32>( cell_uv, 0.0)) * hs;
    let hD = sample_height(uv + vec2<f32>(0.0, -cell_uv)) * hs;
    let hU = sample_height(uv + vec2<f32>(0.0,  cell_uv)) * hs;
    // The cell width in world space.
    let dx = material.domain_size / f32(material.grid_size);
    // Cross-product of the tangents gives the normal.
    let tx = vec3<f32>(2.0 * dx, hR - hL, 0.0);
    let tz = vec3<f32>(0.0, hU - hD, 2.0 * dx);
    return normalize(cross(tz, tx));
}

// ─── Vertex Shader ────────────────────────────────────────────────────────────

@vertex
fn vertex(v: Vertex) -> VertexOut {
    var out: VertexOut;

    // UV maps [−domain/2, +domain/2] → [0, 1].
    out.uv = v.uv;

    // Sample height field.
    let h = sample_height(v.uv) * material.height_scale;

    // Displace vertex upward.
    let local_pos  = vec4<f32>(v.position.x, v.position.y + h, v.position.z, 1.0);
//    let local_pos  = vec4<f32>(v.position.x, v.position.y + h, v.position.z, 1.0);
    let world_pos  = get_world_from_local(v.instance_index) * local_pos;
    out.world_pos  = world_pos.xyz;
    out.clip_pos   = mesh_position_local_to_clip(
        get_world_from_local(v.instance_index),
        local_pos
    );
//    out.clip_pos   = position_world_to_clip(world_pos.xyz);
    out.wave_height = h;
    out.normal     = compute_normal(v.uv);

    return out;
}

// ─── Fragment Shader ──────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> view: View;

// Hardcoded sun direction and colour — replace with your scene's lighting.
const SUN_DIR:   vec3<f32> = vec3<f32>(0.4, 0.8, 0.3);   // normalised in fn
const SUN_COLOR: vec3<f32> = vec3<f32>(1.0, 0.95, 0.8);

@fragment
fn fragment(in: VertexOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
//    return vec4<f32>(in.uv.x, in.uv.y, (in.wave_height/100.0), 0.5);
//    let N = normalize(in.normal);
    let N = normalize(select(-in.normal, in.normal, front_facing));
    let V = normalize(view.world_position.xyz - in.world_pos);
    let L = normalize(SUN_DIR);

    // ── Diffuse / Lambertian ───────────────────────────────────────────────
    let NdotL  = max(dot(N, L), 0.0);
    let diffuse = NdotL * SUN_COLOR;

    // ── Specular (Blinn-Phong approximation, low roughness) ───────────────
    let H       = normalize(L + V);
    let NdotH   = max(dot(N, H), 0.0);
    let gloss   = mix(256.0, 16.0, material.roughness);  // roughness → shininess
    let specular = pow(NdotH, gloss) * SUN_COLOR * 2.0;

    // ── Fresnel (Schlick) ─────────────────────────────────────────────────
    // At grazing angles water is very reflective; at normal incidence it's
    // more transparent.
    let NdotV   = max(dot(N, V), 0.0);
    let F0      = 0.02;   // water at 0° incidence
    let fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    // ── Depth-based colour tinting ────────────────────────────────────────
    // Shallow areas (high wave crest) → lighter turquoise.
    // Deep / trough areas → darker navy.
    let depth_t       = clamp(in.wave_height * 0.4 + 0.5, 0.0, 1.0);
    let shallow_color = vec3<f32>(0.1, 0.6, 0.55);
    let deep_color    = material.base_color.rgb;
    let water_color   = mix(deep_color, shallow_color, depth_t);

    // ── Foam ──────────────────────────────────────────────────────────────
    // Simple threshold on wave height. For a proper foam mask you'd track
    // the Jacobian of the height field in the compute pass.
    let foam_threshold = 0.8 * material.height_scale;
    let foam_amount    = smoothstep(foam_threshold * 0.8, foam_threshold + 100, in.wave_height);
    let foam_color     = vec3<f32>(1.0, 1.0, 1.0);

    // ── Combine ───────────────────────────────────────────────────────────
    var color = water_color * (diffuse * 0.4 + vec3<f32>(0.05));
    color     = mix(color, foam_color, foam_amount);
    color     = color + specular * fresnel;

    // Opacity: more opaque at grazing angles (Fresnel), translucent otherwise.
    var alpha = mix(material.base_color.a, 1.0, fresnel * 0.6 + foam_amount * 0.5);
    alpha = 1.0;

    return vec4<f32>(color, alpha);
}
