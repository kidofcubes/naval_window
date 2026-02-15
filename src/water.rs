use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology, VertexFormat};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError};
use bevy::shader::{ShaderDefVal, ShaderRef};

const SHADER_ASSET_PATH: &str = "custom_material.wgsl";

const ATTRIBUTE_BLEND_COLOR: MeshVertexAttribute =
    MeshVertexAttribute::new("BlendColor", 988540917, VertexFormat::Float32x4);


pub struct WaterPlugin;
impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<CustomMaterial>::default());
        app.add_systems(Startup, setup);
    }
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // let mesh = Mesh::from(Plane3d {
    //         half_size: Vec2::new(50.0, 50.0),
    //         normal: Dir3::Y,
    //     });
    let mesh = generate_dense_plane(Vec3::new(0.0, 1.0, 0.0), 10, 1.0, 1.0, 0.0,0.0);
        // Sets the custom attribute
        // .with_inserted_attribute(
        //     ATTRIBUTE_BLEND_COLOR,
        //     // The cube mesh has 24 vertices (6 faces, 4 vertices per face), so we insert one BlendColor for each
        //     vec![[1.0, 0.0, 0.0, 1.0]; 24],
        // );

    // cube
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(CustomMaterial {
            // color: LinearRgba::BLUE,
            // color_texture: Some(asset_server.load("yuyuchonk.png")),
            // alpha_mode: AlphaMode::Blend,
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}


// This struct defines the data that will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CustomMaterial {
    // #[uniform(0)]
    // color: LinearRgba,
    // #[texture(1)]
//     #[sampler(2)]
//     color_texture: Option<Handle<Image>>,
//     alpha_mode: AlphaMode,
}

/// The Material trait is very configurable, but comes with sensible defaults for all methods.
/// You only need to implement functions for features that need non-default behavior. See the Material api docs for details!
impl Material for CustomMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn specialize(
        pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if descriptor
            .vertex
            .shader_defs
            .contains(&ShaderDefVal::Bool("PREPASS_PIPELINE".into(), true))
        {
            return Ok(());
        }

        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(1),

            // ATTRIBUTE_BLEND_COLOR.at_shader_location(1),

        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }

    // fn alpha_mode(&self) -> AlphaMode {
    //     self.alpha_mode
    // }
}
/// points up in +y
pub fn generate_dense_plane(
    normal: Vec3,
    resolution: u32,
    x_scale: f32,
    z_scale: f32,
    x_offset: f32,
    z_offset: f32,
) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList,RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD);

    let mut vertices : Vec<[f32;3]> = Vec::with_capacity((resolution+1).pow(2) as usize);
    let mut uv: Vec<[f32;2]> = Vec::with_capacity((resolution+1).pow(2) as usize);
    let mut normals: Vec<[f32;3]> = Vec::with_capacity((resolution+1).pow(2) as usize);
    let mut indices: Vec<u32> = Vec::with_capacity(((resolution.pow(2))*2*3) as usize);
    for x in 0..(resolution+1) {
        for z in 0..(resolution+1) {
            vertices.push([
                x as f32 * (x_scale) + x_offset,
                0f32,
                z as f32 * (z_scale) + z_offset
            ]);
            uv.push([
                x as f32 / (resolution as f32),
                z as f32 / (resolution as f32)
            ]);
            normals.push([
                normal.x,
                normal.y,
                normal.z
            ]);
        }
    }
    for x in 0..(resolution) {
        for z in 0..(resolution) {
            //this isn't counter clockwise??? what??? this is clockwise
            indices.push((((x+0)*(resolution+1))+(z+0))); //lower left
            indices.push((((x+0)*(resolution+1))+(z+1))); //upper left
            indices.push((((x+1)*(resolution+1))+(z+0))); //lower right

            indices.push((((x+0)*(resolution+1))+(z+1))); //upper left
            indices.push((((x+1)*(resolution+1))+(z+1))); //upper right
            indices.push((((x+1)*(resolution+1))+(z+0))); //lower right

        }
    }


    // .insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0,0.0]; vertices.len()]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    return mesh

}
