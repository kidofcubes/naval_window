mod cam_movement;
mod water;
mod voxelize;
mod watering;
mod mesh_optimize;
mod big_watering;

use bevy::{color::Color, pbr::wireframe::{WireframeConfig, WireframePlugin}, prelude::*, render::{settings::{RenderCreation, WgpuFeatures, WgpuSettings}, RenderPlugin}, window::WindowResolution};
use std::{env, path::{Path, PathBuf}};
use std::ops::{Deref, DerefMut};
use bevy::asset::UnapprovedPathMode;
use bevy::ecs::event::Trigger;
use bevy::ecs::system::IntoResult;
use bevy::light::{CascadeShadowConfig, CascadeShadowConfigBuilder};
use bevy::math::Affine3A;
use bevy::pbr::DefaultOpaqueRendererMethod;
use bevy::picking::mesh_picking::ray_cast::SimplifiedMesh;
use bevy::render::render_resource::{TextureDimension, TextureFormat};
use naval_sketch::bevy_plugin::{argb_slice_to_color, place_part, BasePartMesh, NavalSketchPlugin};
use naval_sketch::parsing::{load_save};
use naval_sketch::parts::{AdjustableHull, BasePart, Part, PartRegistry};
use naval_sketch::parts_loader::LocalPaths;
// use parry3d::transformation::vhacd::VHACDParameters;
use crate::voxelize::{null_texture, render_sdf, render_voxel_grid, voxelize_meshes, VoxelGrid, VoxelizationConfig};
use crate::cam_movement::CameraMovementPlugin;
use crate::mesh_optimize::{combine_meshes, get_boundary_mesh_conservative, simplify_mesh};
use crate::big_watering::BigWatering;
use crate::water::{CustomMaterial, WaterPlugin};



fn temp_test_update(
    //mut mesh_thing: ResMut<BuildData>,
    mut meshes: ResMut<Assets<Mesh>>,
    key: Res<ButtonInput<KeyCode>>,
    // mut query: Query<(Entity, &mut Mesh3d, &mut MeshMaterial3d<StandardMaterial>, &BasePartMesh)>,
    all_renderers: Query<(&Mesh3d, &GlobalTransform, Entity), With<BasePartMesh>>,
    hull_query: Query<(Entity, &BasePart, &mut AdjustableHull)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut build_data: ResMut<BuildData>,
    mut light: Query<(Entity, &DirectionalLight)>,
    scenes: ResMut<Assets<Scene>>,
    parent_query: Query<&ChildOf>,
    base_part_query: Query<(Entity, &BasePart)>,
    mut visibility_query: Query<(Entity, &mut Visibility)>,
    mut transform_query: Query<&mut Transform>,
    mut commands: Commands,
    mut gizmos: Gizmos
) {
    // if key.just_pressed(KeyCode::KeyH) {
    //     let hovered_mat = StandardMaterial::from_color(Color::WHITE);
    //     let hovered_mat_handle = materials.add(hovered_mat);
    //
    //
    //     for mut pair in &mut query {
    //         //let colored_mat = StandardMaterial::from_color(base_part_query.get(parent_query.root_ancestor(pair.0)).unwrap().1.color);
    //
    //         let colored_mat = StandardMaterial::from_color(argb_slice_to_color(&base_part_query.get(pair.3.base_part).unwrap().1.color));
    //         let colored_mat_handle = materials.add(colored_mat);
    //         // materials.insert(pair.2.id(),
    //         //     colored_mat.clone()
    //         // );
    //         pair.2.0 = colored_mat_handle.clone();
    //         // commands.entity(pair.0)
    //         //     .observe(update_material_on::<Pointer<Over>>(hovered_mat_handle.clone()))
    //         //     .observe(update_material_on::<Pointer<Out>>(colored_mat_handle))
    //         // ;
    //     }
    // }
    if key.pressed(KeyCode::KeyR) {
        let mut transform = transform_query.get_mut(build_data.root_ship_entity.unwrap()).unwrap();
        let mut orig_rotation = transform.rotation.to_euler(EulerRot::XYZ);
        orig_rotation.1 += 0.01;
        println!("rotation {:?}",orig_rotation);
        transform.rotation = Quat::from_euler(EulerRot::XYZ, orig_rotation.0, orig_rotation.1, orig_rotation.2);
    }
    if key.pressed(KeyCode::KeyQ){
        let mut total_triangles = 0;
        for mesh_handle in all_renderers.iter() {
            if let Some(mesh) = meshes.get(mesh_handle.0) && let Some(indices) = mesh.indices(){
                total_triangles += indices.len() / 3;
            }
        }
        info!("TOTAL TRIANGLES {}",total_triangles);
    }
    
    // Test voxelization functionality
    if key.just_pressed(KeyCode::KeyV) {
        info!("Running voxelization test...");
        
        // Collect all mesh references for voxelization
        let mesh_refs: Vec<(&Mesh, &GlobalTransform)> = all_renderers
            .iter()
            .filter_map(|x| Some((meshes.get(x.0).unwrap(), x.1)))
            .collect();
            
        if !mesh_refs.is_empty() {
            // Configure voxelization
            let config = VoxelizationConfig {
                resolution: 64,
                min_voxel_size: Some(0.25),
                max_voxel_size: Some(1.0),
            };
            
            // Voxelize the meshes
            let voxel_grid = voxelize_meshes(mesh_refs, config);
            info!("Created voxel grid with dimensions: {:?}", voxel_grid.dimensions);
            info!("Voxel size: {}", voxel_grid.voxel_size);
            build_data.voxel_grid = Some(voxel_grid);
        } else {
            warn!("No meshes found for voxelization");
        }
    }

    if(key.pressed(KeyCode::KeyB)){
        if let Some(voxel_grid) = &build_data.voxel_grid {
            render_voxel_grid(&voxel_grid, &mut gizmos);
            info!("kjasjfklajdfjd {}",voxel_grid.voxels.len());
        }
    }

    if(key.pressed(KeyCode::KeyN)){
        if let Some(voxel_grid) = &build_data.voxel_grid {
            render_sdf(&voxel_grid, &mut gizmos);
        }
    }

    if(key.just_pressed(KeyCode::KeyJ)){
        let mesh_list: Vec<(&Mesh, &GlobalTransform)>= all_renderers.iter().map(|x| (meshes.get(x.0).unwrap(), x.1)).collect();
        let combined_mesh = combine_meshes(&mesh_list).unwrap();
        println!("orig_tris {:?}",combined_mesh.count_vertices()/3);
        println!("indices {:?}",combined_mesh.indices().is_some());
        let mut new_mesh: Option<Mesh> = None;

        if let Some(voxel_grid) = &build_data.voxel_grid {
            new_mesh = Some(simplify_mesh(&combined_mesh, 0.005).unwrap());
            // println!("no culling new_mesh {:?}",new_mesh.unwrap().count_vertices()/3);

            // new_mesh = get_boundary_mesh_conservative(voxel_grid, combined_mesh, &GlobalTransform::default());
            // new_mesh = Some(simplify_mesh(&new_mesh.unwrap(), 0.01).unwrap());
            // new_mesh = Some(combined_mesh);
        }



        // new_mesh = Some(simplify_mesh(&combined_mesh, 0.01).unwrap());
        // let new_meshes: Vec<Mesh> = convex_decomposition(&combined_mesh, Some(VHACDParameters {
        //     concavity: 0.0001,
        //     resolution: 512,
        //     max_convex_hulls: 16384,
        //     ..default()
        // })).unwrap();
        // let new_meshes_with_transform = new_meshes.iter().map(|x| (x, &GlobalTransform::IDENTITY)).collect::<Vec<_>>();
        // let new_mesh = combine_meshes(&new_meshes_with_transform).unwrap();
        if let Some(new_mesh) = new_mesh {
            println!("tris {:?}",new_mesh.count_vertices()/3);

            let mesh_entity = commands.spawn((
                Mesh3d(meshes.add(new_mesh)),
                MeshMaterial3d(materials.add(StandardMaterial::from_color(Color::srgba_u8(255,255,255,128)))),
                Transform::from_xyz(0.0, 0.0, 0.0),
            ));
            build_data.simplified_mesh = Some(mesh_entity.id());
        }

        // build_data.root_ship_entity
    }

    if(key.just_pressed(KeyCode::KeyK)){

        for (entity) in all_renderers.iter().map(|x| x.2) {

            let new_hidden = if let Visibility::Hidden = visibility_query.get(entity).unwrap().1 {Visibility::Visible} else {Visibility::Hidden};
            // *visibility_query.get_mut(entity).unwrap().1 = Visibility::Hidden;
            *visibility_query.get_mut(entity).unwrap().1 = new_hidden;
        }
        if let Ok(mut visibility) = visibility_query.get_mut(build_data.root_ship_entity.unwrap()) {
            // visibility.1.toggle_visible_hidden();
            // let simplified_hidden = if let Visibility::Hidden = visibility.1.deref() {true} else {false};
            // let mut base_ship = visibility_query.get_mut(build_data.root_ship_entity.unwrap()).unwrap().1;
            // if let Visibility::Hidden = base_ship.deref() {
            //     if(simplified_hidden) {
            //         base_ship.toggle_visible_hidden();
            //     }
            // } else {
            //     if(!simplified_hidden) {
            //         base_ship.toggle_visible_hidden();
            //     }
            // }
            // for (_, _, mut visibility) in all_renderers.iter_mut() {
            //     visibility.toggle_visible_hidden();
            // }
        }
    }
    if(key.just_pressed(KeyCode::KeyL)){
        let mut transform = transform_query.get_mut(light.single().unwrap().0).unwrap();
        transform.translation = Vec3::new(25.0, 5.0, -20.5) * 10000.0;
        transform.rotation = transform.looking_at(Vec3 {x:0.0, y:0.0, z:0.0 }, Vec3::Y).rotation;
    }
}







#[derive(Resource)]
struct InitData {
    file_path: Option<String>,
    data_paths: Option<LocalPaths>
}

#[derive(Resource)]
pub struct BuildData{
    pub root_ship_entity: Option<Entity>,
    pub voxel_grid: Option<VoxelGrid>,
    //reference to the entity that has the mesh
    pub simplified_mesh: Option<Entity>,
}




/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut build_data: ResMut<BuildData>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    init_data: Option<Res<InitData>>,
    asset_server: Res<AssetServer>,
    part_registry: Option<Res<PartRegistry>>,
    //mut scene_assets: ResMut<Assets<Scene>>,
    mut ambient_light: ResMut<GlobalAmbientLight>,
    mut config_store: ResMut<GizmoConfigStore>,
) -> Result<()>{

    // let path = ;

    if let Some(path) = init_data.and_then(|x| x.file_path.clone()) && let Some(part_registry) = part_registry{
        let parts_result = load_save(Path::new(&path));
        build_data.root_ship_entity = Some(commands.spawn((
            Transform::default(),
            InheritedVisibility::VISIBLE
        )).id());


        println!("PLACING PARTS");
        if let Ok(parts) = parts_result {
            for part in parts {
                let mut entity = commands.spawn_empty();
                entity.insert(ChildOf(build_data.root_ship_entity.unwrap()));
                // entity.set_parent_in_place(build_data.root_ship_entity.unwrap());
                if let Ok(_) = place_part(
                    &mut meshes,
                    &mut materials,
                    &asset_server,
                    &part_registry,
                    &mut entity,
                    &part) {
                }else {
                    println!("ERROR PLACING PART {:?}",part);
                }
            }
        }else{
            println!("ERROR WAS {:?}",parts_result);
            place_part(
                &mut meshes,
                &mut materials,
                &asset_server,
                &part_registry,
                &mut commands.spawn_empty(),
                &Part::Normal(BasePart {
                    id: 5,
                    ignore_physics: false,
                    position: Vec3 {x:10.0,y:10.0,z:10.0},
                    rotation: Vec3::ZERO,
                    scale: Vec3 {x:5.0,y:1.0,z:5.0},
                    color: [255, 255, 255, 255],
                    armor: 0,
                }))?;


            place_part(
                &mut meshes,
                &mut materials,
                &asset_server,
                &part_registry,
                &mut commands.spawn_empty(),
                &Part::Normal(BasePart {
                    id: 5,
                    ignore_physics: false,
                    position: Vec3 {x:20.0,y:10.0,z:10.0},
                    rotation: Vec3::ZERO,
                    scale: Vec3 {x:1.0,y:1.0,z:1.0},
                    color: [255, 0, 255, 0],
                    armor: 0,
                }))?;


            // commands.spawn((
            //     Mesh3d(meshes.add(Cuboid::default())),
            //     MeshMaterial3d(materials.add(Color::srgb_u8(0, 0, 255))),
            //     Transform::from_translation(Vec3::new(0.0,30.0,0.0))
            // ));
            // commands.spawn((
            //     Mesh3d(meshes.add(Cuboid::default())),
            //     MeshMaterial3d(materials.add(Color::srgb_u8(255, 0, 0))),
            //     Transform::from_translation(Vec3::new(0.0,30.0,1.0))
            // ));




        }
    }


    let mut light_transform = Transform::from_xyz(2500.0, 500.0, -2500.0);
    light_transform = light_transform.looking_at(Vec3 {x:0.0, y:0.0, z:0.0 }, Vec3::Y);
    light_transform.translation = Vec3::new(25.0, 5.0, -20.5) * 10000.0;
    light_transform.rotation = light_transform.looking_at(Vec3 {x:0.0, y:0.0, z:0.0 }, Vec3::Y).rotation;
    // light
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            color: Color::srgb(1.0, 0.83, 0.71),
            shadows_enabled: true,
            // soft_shadow_size: None,
            ..default()
        },
        CascadeShadowConfigBuilder {
            maximum_distance: 8192.0,
            // minimum_distance: 8.0,
            // first_cascade_far_bound: 16.0,
            num_cascades: 4,
            // overlap_proportion: 0.05,
            ..default()
        }.build(),
        light_transform,
    ));
    ambient_light.brightness = 00.0;


    //commands.spawn();

    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();

    config.depth_bias = -1.0;



    return Ok(());



}

//NOTE, SCALE IS THE FULL WIDTH

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Error);
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        // futures::executor::block_on(get_all_parts(None));
        // .await;
    }


    let args: Vec<String> = env::args().collect();

    let mut data_paths: Option<LocalPaths> = None;
    let mut file_path: Option<String> = None;

    #[cfg(not(target_arch = "wasm32"))]
    {
        let steam_path = &args[1];
        file_path = Some(args[2].clone());


        let cache_folder = dirs::cache_dir().unwrap().join("naval_sketch");
        let steam_folder = PathBuf::from(steam_path.clone());
        let workshop_folder = steam_folder.join("steamapps").join("workshop").join("content").join("842780");
        let game_folder = steam_folder.join("steamapps").join("common").join("NavalArt");
        std::fs::create_dir_all(&cache_folder).unwrap();


        data_paths = Some(LocalPaths {
            cache_folder,
            game_folder,
            workshop_folder
        });
    }


    let mut default_plugins = DefaultPlugins.set(RenderPlugin {
        render_creation: RenderCreation::Automatic(WgpuSettings {
            // WARN this is a native only feature. It will not work with webgl or webgpu
            // features: WgpuFeatures::POLYGON_MODE_LINE,
            ..default()
        }),
        ..default()
    }).set(
        AssetPlugin {
            watch_for_changes_override: Some(true),
            mode: AssetMode::Unprocessed,
            meta_check: bevy::asset::AssetMetaCheck::Never,
            unapproved_path_mode: UnapprovedPathMode::Deny,
            ..default()
        }
    );

    #[cfg(target_arch = "wasm32")]
    {
        default_plugins = default_plugins.set(
            WindowPlugin {
                primary_window: Some(Window {
                    // provide the ID selector string here
                    canvas: Some("#game-canvas".into()),
                    // ... any other window properties ...
                    ..default()
                }),
                ..default()
            }
        )
    }


    App::new()
        .insert_resource(InitData {
            file_path,
            data_paths: data_paths.clone()
        })
        .insert_resource(ClearColor(Color::srgba_u8(0,0,0,0)))
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(WireframeConfig {
            // The global wireframe config enables drawing of wireframes on every mesh,
            // except those with `NoWireframe`. Meshes with `Wireframe` will always have a wireframe,
            // regardless of the global configuration.
            global: false,
            // Controls the default color of all wireframes. Used as the default color for global wireframes.
            // Can be changed per mesh using the `WireframeColor` component.
            default_color: Color::WHITE,
        })
        .insert_resource(BuildData{
            root_ship_entity: None,
            voxel_grid: None,
            simplified_mesh: None,
        })
        .add_plugins((
            // WebAssetPlugin,
            default_plugins,
            NavalSketchPlugin{
                init_paths: data_paths.unwrap(),
            },
            WireframePlugin::default(),
            WaterPlugin,
            CameraMovementPlugin,
            MeshPickingPlugin,
            // BigWatering,
            //OutlinePlugin,
        ))
        .add_systems(Startup, (setup).chain())
        .add_systems(Update, (temp_test_update))


        .run();

}
