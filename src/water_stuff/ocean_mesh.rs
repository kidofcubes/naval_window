use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup},
};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::render::render_resource::{RenderPipelineDescriptor, SpecializedMeshPipelineError};
use bevy::shader::{ShaderDefVal, ShaderRef};
use super::{HeightFieldHandle, DOMAIN_SIZE, GRID_SIZE};

// ─── Water Material ───────────────────────────────────────────────────────────

/// Standard PBR-style water material that samples the computed height field
/// in the vertex shader and uses analytical normals in the fragment shader.
///
/// Bind group 0/1/2 are occupied by Bevy's standard view + mesh bindings.
/// The height field is bound at group 3 (our custom group).
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct WaterMaterial {
    /// Base water colour (deep colour; shallow colour is derived from depth).
    #[uniform(0)]
    pub base_color:      Vec4,
    /// Roughness for PBR specular (0 = mirror, 1 = fully rough).
    #[uniform(0)]
    pub roughness:       f32,
    /// Height scale applied in vertex shader (visual amplitude multiplier).
    #[uniform(0)]
    pub height_scale:    f32,
    /// Size of the mesh in world-space metres (should match DOMAIN_SIZE).
    #[uniform(0)]
    pub domain_size:     f32,
    /// Simulation grid resolution.
    #[uniform(0)]
    pub grid_size:       u32,

    /// The reconstructed height field texture (R32Float, GRID_SIZE × GRID_SIZE).
    /// Automatically inserted by `spawn_water_mesh`.
    #[texture(1)]
    #[sampler(2)]
    pub height_field:    Handle<Image>,
}

impl Material for WaterMaterial {
    fn vertex_shader() -> ShaderRef {
        "water_surface.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "water_surface.wgsl".into()
    }
    // Water is translucent, so render in the transparent pass.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
    fn depth_bias(&self) -> f32 {
        return 0.0;
    }



    // fn enable_shadows() -> bool {
    //     return true;
    // }
    // fn enable_prepass() -> bool {
    //     return true;
    // }

    fn specialize(pipeline: &MaterialPipeline, descriptor: &mut RenderPipelineDescriptor, layout: &MeshVertexBufferLayoutRef, key: MaterialPipelineKey<Self>) -> Result<(), SpecializedMeshPipelineError> {
        // Explicitly enable depth write even in transparent pass
        if let Some(depth_stencil) = descriptor.depth_stencil.as_mut() {
            depth_stencil.depth_write_enabled = true;
        }

        descriptor.primitive.cull_mode = None;

        if descriptor
            .vertex
            .shader_defs
            .contains(&ShaderDefVal::Bool("PREPASS_PIPELINE".into(), true))
        {
            return Ok(());
        }

        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),

            // ATTRIBUTE_BLEND_COLOR.at_shader_location(1),

        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

// ─── Marker Component ─────────────────────────────────────────────────────────

#[derive(Component)]
pub struct WaterMesh;

// ─── Spawn Helper ─────────────────────────────────────────────────────────────

/// Call once from your setup system.
///
/// ```rust
/// fn setup(mut commands: Commands, ...) {
///     spawn_water_mesh(&mut commands, &mut meshes, &mut water_materials, &mut images);
/// }
/// ```
pub fn spawn_water_mesh(
    commands:        &mut Commands,
    meshes:          &mut Assets<Mesh>,
    materials:       &mut Assets<WaterMaterial>,
    images:          &mut Assets<Image>,
    hf_handle:       &Res<HeightFieldHandle>,
) -> Entity {
    use bevy::render::render_resource::{
        Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    };

    // Create a placeholder height field image.  The actual data is written
    // each frame by the compute shader, so this just needs the right
    // format and size to allocate GPU memory.;
    // let hf_handle = images.add(hf_image);
    // let hf_handle = images.add(hf_image);
    // let hf_handle = hf_handle;

    // Tessellated plane.  Higher subdivisions = smoother wave geometry.
    // 256×256 gives good quality; reduce to 128 for lower-end hardware.
    let subdivisions: u32 = 256;
    let mesh = plane_mesh(DOMAIN_SIZE, subdivisions);
    let mesh_handle = meshes.add(mesh);

    let mat = materials.add(WaterMaterial {
        // base_color:   vec4(0.05, 0.3, 0.5, 0.85),
        base_color:   vec4(0.05, 0.3, 0.5, 1.0),
        roughness:    0.05,
        height_scale: 1.5,
        domain_size:  DOMAIN_SIZE,
        grid_size:    GRID_SIZE,
        height_field: hf_handle.0.clone(),
    });

    commands.spawn((
        WaterMesh,
        Mesh3d(mesh_handle),
        MeshMaterial3d(mat),
        Transform::IDENTITY
    )).id()
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build a flat XZ plane mesh with `subdivisions` quads per axis.
fn plane_mesh(size: f32, subdivisions: u32) -> Mesh {
    let verts_per_axis = (subdivisions + 1) as usize;
    let total_verts    = verts_per_axis * verts_per_axis;

    let mut positions  = Vec::with_capacity(total_verts);
    let mut normals    = Vec::with_capacity(total_verts);
    let mut uvs        = Vec::with_capacity(total_verts);
    let mut indices    = Vec::with_capacity((subdivisions * subdivisions * 6) as usize);

    let half = size * 0.5;
    let step = size / subdivisions as f32;

    for j in 0..=subdivisions {
        for i in 0..=subdivisions {
            let x = -half + i as f32 * step;
            let z = -half + j as f32 * step;
            positions.push([x, 0.0, z]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([i as f32 / subdivisions as f32, j as f32 / subdivisions as f32]);
        }
    }

    for j in 0..subdivisions {
        for i in 0..subdivisions {
            let tl = j * (subdivisions + 1) + i;
            let tr = tl + 1;
            let bl = tl + (subdivisions + 1);
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::all(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0,     uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
