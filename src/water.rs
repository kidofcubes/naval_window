use std::ops::Add;
use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::Aabb;
use bevy::light::NotShadowCaster;
// use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology, VertexFormat};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPipeline, MaterialPipelineKey, OpaqueRendererMethod};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError, TextureDimension, TextureFormat};
use bevy::shader::{ShaderDefVal, ShaderRef};
use rand::{random, random_bool, random_range};
use crate::{watering, BuildData};
use crate::voxelize::{create_voxel_texture, null_texture, VoxelGrid};
use crate::watering::{FftOceanConfig, FftOceanHandles, HeightFieldHandle, VoxelGridHandle, VoxelSettings, Watering, WaterSimulationHandles};
use crate::watering::water_lod::WaterLodPlugin;

const SHADER_ASSET_PATH: &str = "custom_material.wgsl";

const ATTRIBUTE_BLEND_COLOR: MeshVertexAttribute =
    MeshVertexAttribute::new("BlendColor", 988540917, VertexFormat::Float32x4);


pub struct WaterPlugin;
impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<ExtendedMaterial<StandardMaterial, CustomMaterial>>{
            ..default()
        });

        app.add_plugins(Watering);
        app.add_plugins(WaterLodPlugin {});
        app.add_systems(Startup, setup.after(watering::setup_textures));
        app.add_systems(Update, update);
    }
}


/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut extended_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, CustomMaterial>>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut height_field_handle: ResMut<HeightFieldHandle>,
    water_sim_handles: Res<WaterSimulationHandles>,
    fft_handles: Res<FftOceanHandles>,
    fft_config: Res<FftOceanConfig>,
    asset_server: Res<AssetServer>,
) {

    // let mesh = Mesh::from(Plane3d {
    //         half_size: Vec2::new(50.0, 50.0),
    //         normal: Dir3::Y,
    //     });
    // let mesh = generate_dense_plane(Vec3::new(0.0, 1.0, 0.0), 1024, 400.0, 400.0, -200.0,-200.0);
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


    let water_lod_mesh = crate::watering::water_lod::WaterLodMesh {
        last_camera_pos: Vec3::ZERO,
        center: Vec2::ZERO,
    };

    let mesh     = crate::watering::water_lod::build_lod_mesh(Vec3::ZERO, &water_lod_mesh);
    //
    // // ── Replace this material with your custom water shader material ──────────
    // let material = materials.add(StandardMaterial {
    //     base_color:           Color::srgb(0.04, 0.28, 0.46),
    //     perceptual_roughness: 0.05,
    //     reflectance:          0.9,
    //     ..default()
    // });

    let custom_material = CustomMaterial {
        time: 0.0,
        voxel_transform: Mat4::IDENTITY,
        voxel_size: 0.5,
        voxel_grid: null_texture(images.as_mut(), TextureDimension::D3, TextureFormat::R8Uint),
        amplitude_texture: water_sim_handles.amplitude_texture_a.clone(),
        profile_buffer: water_sim_handles.profile_buffer.clone(),
        display_mode: 1, // default to FFT ocean
        fft_height_0: fft_handles.cascades[0].height_field.clone(),
        fft_slope_0:  fft_handles.cascades[0].slope_field.clone(),
        fft_height_1: fft_handles.cascades[1].height_field.clone(),
        fft_slope_1:  fft_handles.cascades[1].slope_field.clone(),
        fft_height_2: fft_handles.cascades[2].height_field.clone(),
        fft_slope_2:  fft_handles.cascades[2].slope_field.clone(),
        fft_patch_lengths: Vec3::new(
            fft_config.patch_lengths[0],
            fft_config.patch_lengths[1],
            fft_config.patch_lengths[2],
        ),
    };

    let material = ExtendedMaterial {
        base: StandardMaterial {
            // base_color: Color::srgb_u8(26, 100, 200),
            // base_color: Color::srgb_u8(7, 48, 71),
            base_color: Color::srgb(0.003, 0.008, 0.015),
            opaque_render_method: OpaqueRendererMethod::Deferred,
            perceptual_roughness: 0.25,
            reflectance: 0.4,
            metallic: 0.0,
            alpha_mode: AlphaMode::Opaque,
            ..default()
        },
        extension: custom_material
    };


    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(extended_materials.add(material)),
        // NoFrustumCulling,
        Transform::from_xyz(0.0, -100.0, 0.0),
        water_lod_mesh,
        NotShadowCaster
    ));
    // commands.insert_resource(wake_sim);
}

fn update(
    // mut materials: ResMut<Assets<CustomMaterial>>,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, CustomMaterial>>>,
    mut build_data: ResMut<BuildData>,
    mut images: ResMut<Assets<Image>>,
    key: Res<ButtonInput<KeyCode>>,
    mut voxel_grid_handle: ResMut<VoxelGridHandle>,
    mut voxel_settings: ResMut<VoxelSettings>,
    mut sim_handles: ResMut<WaterSimulationHandles>,
    time: Res<Time>,
    mut gizmo: Gizmos,
){
    sim_handles.flipped = !sim_handles.flipped;
    for (_, mut material) in materials.iter_mut() {
        material.extension.time = (time.elapsed().as_millis() as f32 / 1000.0) as f32;
        if let Some(voxel_grid) = build_data.voxel_grid.as_ref() && key.just_pressed(KeyCode::KeyG) {
            update_material(&mut material.extension, voxel_grid, &mut images)
        }
        material.extension.amplitude_texture = if sim_handles.flipped {
            sim_handles.amplitude_texture_b.clone()
        } else {
            sim_handles.amplitude_texture_a.clone()
        };

        // M key toggles display mode between wavelets (0) and FFT ocean (1)
        if key.just_pressed(KeyCode::KeyM) {
            material.extension.display_mode = if material.extension.display_mode == 0 { 1 } else { 0 };
            println!("Display mode: {}", if material.extension.display_mode == 0 { "wavelets" } else { "FFT ocean" });
        }
    }
    // if let Some(voxel_grid) = build_data.voxel_grid.as_ref() && key.just_pressed(KeyCode::KeyG) {
    //     voxel_grid_handle.0 = create_voxel_texture(voxel_grid,&mut images);
    //     voxel_settings.voxel_size = voxel_grid.voxel_size;
    //     voxel_settings.water_to_world = Mat4::from_translation(voxel_grid.offset.into()).mul_mat4(&Mat4::from_scale(Vec3::splat(voxel_grid.voxel_size))).inverse();
    //     voxel_settings.grid_dims = uvec3(voxel_grid.dimensions.0,voxel_grid.dimensions.1,voxel_grid.dimensions.2);
    // }
}
fn update_material(mut material: &mut CustomMaterial, voxel_grid: &VoxelGrid, images: &mut Assets<Image>){
    println!("thing is {}",material.voxel_transform.mul_vec4(vec4(1.0,1.0,1.0,1.0)));
    material.voxel_grid = create_voxel_texture(voxel_grid,images);
    material.voxel_size = voxel_grid.voxel_size;
    // material.voxel_transform = Mat4::from_translation(voxel_grid.offset.into()).mul_mat4(&Mat4::from_scale(Vec3::splat(voxel_grid.voxel_size))).inverse();
    // material.voxel_transform = Mat4::from_translation((-voxel_grid.offset).into()).mul_mat4(&Mat4::from_scale(Vec3::splat(1.0/voxel_grid.voxel_size)));
    material.voxel_transform = Mat4::from_scale(Vec3::splat(1.0/voxel_grid.voxel_size)).mul_mat4(&Mat4::from_translation((-voxel_grid.offset).into()));
    // material.voxel_transform = Mat4::from_translation((-voxel_grid.offset).into());
    // material.voxel_transform = Mat4::from_translation(voxel_grid.offset.into());
}

// This struct defines the data that will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CustomMaterial {
    #[uniform(100)]
    time: f32,
    #[uniform(101)]
    voxel_transform: Mat4,
    #[uniform(102)]
    voxel_size: f32,
    #[texture(103, dimension = "3d", sample_type = "u_int")]
    voxel_grid: Handle<Image>,

    #[texture(109, dimension = "2d_array", sample_type = "float")]
    #[sampler(110)]
    amplitude_texture: Handle<Image>,

    #[texture(111, dimension = "2d", sample_type = "float")]
    profile_buffer: Handle<Image>,

    #[uniform(112)]
    display_mode: u32, // 0 = wavelets, 1 = FFT ocean

    // Cascade 0
    #[texture(113, dimension = "2d", sample_type = "float")]
    fft_height_0: Handle<Image>,
    #[texture(114, dimension = "2d", sample_type = "float")]
    #[sampler(115)]
    fft_slope_0: Handle<Image>,

    // Cascade 1
    #[texture(116, dimension = "2d", sample_type = "float")]
    fft_height_1: Handle<Image>,
    #[texture(117, dimension = "2d", sample_type = "float")]
    fft_slope_1: Handle<Image>,

    // Cascade 2
    #[texture(118, dimension = "2d", sample_type = "float")]
    fft_height_2: Handle<Image>,
    #[texture(119, dimension = "2d", sample_type = "float")]
    fft_slope_2: Handle<Image>,

    #[uniform(120)]
    fft_patch_lengths: Vec3,
}

/// The Material trait is very configurable, but comes with sensible defaults for all methods.
/// You only need to implement functions for features that need non-default behavior. See the Material api docs for details!
impl MaterialExtension for CustomMaterial {
    // fn vertex_shader() -> ShaderRef {
    //     SHADER_ASSET_PATH.into()
    // }
    fn prepass_vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
    fn deferred_vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    // fn fragment_shader() -> ShaderRef {
    //     ShaderRef::Default
    // }
    fn deferred_fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
        // ShaderRef::Default
    }
    // fn prepass_fragment_shader() -> ShaderRef {
    //     // SHADER_ASSET_PATH.into()
    //     ShaderRef::Default
    // }

    // fn fragment_shader() -> ShaderRef {
    //     SHADER_ASSET_PATH.into()
    // }
    //
    // fn prepass_fragment_shader() -> ShaderRef {
    //     // SHADER_ASSET_PATH.into()
    //     ShaderRef::Default
    // }
    //
    // fn fragment_shader() -> ShaderRef {
    //     // SHADER_ASSET_PATH.into()
    //     ShaderRef::Default
    // }

    // fn specialize(
    //     pipeline: &MaterialPipeline,
    //     descriptor: &mut RenderPipelineDescriptor,
    //     layout: &MeshVertexBufferLayoutRef,
    //     key: MaterialPipelineKey<Self>,
    // ) -> Result<(), SpecializedMeshPipelineError> {
    //     if descriptor
    //         .vertex
    //         .shader_defs
    //         .contains(&ShaderDefVal::Bool("PREPASS_PIPELINE".into(), true))
    //     {
    //         return Ok(());
    //     }
    //
    //     let vertex_layout = layout.0.get_layout(&[
    //         Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
    //         Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
    //
    //         // ATTRIBUTE_BLEND_COLOR.at_shader_location(1),
    //
    //     ])?;
    //     descriptor.vertex.buffers = vec![vertex_layout];
    //     // descriptor.primitive.cull_mode = None;
    //     Ok(())
    // }

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
