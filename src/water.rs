use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::Aabb;
// use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology, VertexFormat};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError};
use bevy::shader::{ShaderDefVal, ShaderRef};
use crate::BuildData;
use crate::voxelize::{create_voxel_texture, null_texture, VoxelGrid};

const SHADER_ASSET_PATH: &str = "custom_material.wgsl";

const ATTRIBUTE_BLEND_COLOR: MeshVertexAttribute =
    MeshVertexAttribute::new("BlendColor", 988540917, VertexFormat::Float32x4);


pub struct WaterPlugin;
impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<CustomMaterial>{
            ..default()
        });
        app.add_systems(Startup, setup);
        app.add_systems(Update, update);
    }
}


/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut custom_materials: ResMut<Assets<CustomMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // let mesh = Mesh::from(Plane3d {
    //         half_size: Vec2::new(50.0, 50.0),
    //         normal: Dir3::Y,
    //     });
    let mesh = generate_dense_plane(Vec3::new(0.0, 1.0, 0.0), 100, 1000.0, 1000.0, -500.0,-500.0);
        // Sets the custom attribute
        // .with_inserted_attribute(
        //     ATTRIBUTE_BLEND_COLOR,
        //     // The cube mesh has 24 vertices (6 faces, 4 vertices per face), so we insert one BlendColor for each
        //     vec![[1.0, 0.0, 0.0, 1.0]; 24],
        // );

    // cube
    // commands.spawn((
    //     Mesh3d(meshes.add(Cuboid::default())),
    //     MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
    //     Transform::from_xyz(0.0, 0.0, 0.0),
    // ));


    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(custom_materials.add(CustomMaterial {
            // color: LinearRgba::BLUE,
            // color_texture: Some(asset_server.load("yuyuchonk.png")),
            // alpha_mode: AlphaMode::Blend,
            time: 0.0,
            voxel_transform: Mat4::IDENTITY,
            voxel_size: 0.5,
            voxel_grid: null_texture(images.as_mut())
        })),
        // NoFrustumCulling,
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

fn update(
    mut materials: ResMut<Assets<CustomMaterial>>,
    mut build_data: ResMut<BuildData>,
    mut images: ResMut<Assets<Image>>,
    key: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
){
    for (_, mut material) in materials.iter_mut() {
        // material.time = (time.elapsed().as_nanos() / (1000u128 * 1000u128 * 1000u128)) as f32;
        material.time = (time.elapsed().as_millis() as f32 / 1000.0) as f32;
        if let Some(voxel_grid) = build_data.voxel_grid.as_ref() && key.pressed(KeyCode::KeyG) {
            update_material(&mut material, voxel_grid, &mut images)
        }
    }
}
fn update_material(mut material: &mut CustomMaterial, voxel_grid: &VoxelGrid, images: &mut Assets<Image>){
    println!("thing is {}",material.voxel_transform.mul_vec4(vec4(1.0,1.0,1.0,1.0)));
    material.voxel_grid = create_voxel_texture(voxel_grid,images);
    material.voxel_size = voxel_grid.voxel_size;
    material.voxel_transform = Mat4::from_translation(voxel_grid.offset.into()).mul_mat4(&Mat4::from_scale(Vec3::splat(voxel_grid.voxel_size))).inverse();
    // material.voxel_transform = Mat4::from_translation(voxel_grid.offset.into());
}

// This struct defines the data that will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CustomMaterial {
    #[uniform(0)]
    time: f32,
    #[uniform(1)]
    voxel_transform: Mat4,
    #[uniform(2)]
    voxel_size: f32,
    #[texture(3, dimension = "3d", sample_type = "u_int")]
    // #[sampler(3)]
    voxel_grid: Handle<Image>,
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
    // fn prepass_vertex_shader() -> ShaderRef {
    //     SHADER_ASSET_PATH.into()
    // }
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
        descriptor.primitive.cull_mode = None;
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
                (x as f32 * (x_scale/(resolution as f32))) + x_offset,
                -100.0f32,
                (z as f32 * (z_scale/(resolution as f32))) + z_offset
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
