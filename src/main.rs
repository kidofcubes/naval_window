mod cam_movement;
mod water;

use bevy::{color::Color, pbr::wireframe::{WireframeConfig, WireframePlugin}, prelude::*, reflect::List, render::{settings::{RenderCreation, WgpuFeatures, WgpuSettings}, RenderPlugin}, window::WindowResolution};
use std::{env, path::{Path, PathBuf}};
use bevy::asset::UnapprovedPathMode;
use bevy::ecs::event::Trigger;
use naval_sketch::bevy_plugin::{argb_slice_to_color, place_part, BasePartMesh, NavalSketchPlugin};
use naval_sketch::parsing::{load_save};
use naval_sketch::parts::{AdjustableHull, BasePart, Part, PartRegistry};
use naval_sketch::parts_loader::LocalPaths;
use crate::cam_movement::CameraMovementPlugin;
use crate::water::WaterPlugin;

fn temp_test_update(
    //mut mesh_thing: ResMut<BuildData>,
    meshes: ResMut<Assets<Mesh>>,
    key: Res<ButtonInput<KeyCode>>,
    mut query: Query<(Entity, &mut Mesh3d, &mut MeshMaterial3d<StandardMaterial>, &BasePartMesh)>,
    hull_query: Query<(Entity, &BasePart, &mut AdjustableHull)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut build_data: ResMut<BuildData>,
    scenes: ResMut<Assets<Scene>>,
    parent_query: Query<&ChildOf>,
    base_part_query: Query<(Entity, &BasePart)>,
    mut transform_query: Query<&mut Transform>,
    commands: Commands,
    gizmo: Gizmos
) {
    if key.just_pressed(KeyCode::KeyH) {
        let hovered_mat = StandardMaterial::from_color(Color::WHITE);
        let hovered_mat_handle = materials.add(hovered_mat);


        for mut pair in &mut query {
            //let colored_mat = StandardMaterial::from_color(base_part_query.get(parent_query.root_ancestor(pair.0)).unwrap().1.color);

            let colored_mat = StandardMaterial::from_color(argb_slice_to_color(&base_part_query.get(pair.3.base_part).unwrap().1.color));
            let colored_mat_handle = materials.add(colored_mat);
            // materials.insert(pair.2.id(),
            //     colored_mat.clone()
            // );
            pair.2.0 = colored_mat_handle.clone();
            // commands.entity(pair.0)
            //     .observe(update_material_on::<Pointer<Over>>(hovered_mat_handle.clone()))
            //     .observe(update_material_on::<Pointer<Out>>(colored_mat_handle))
            // ;
        }
    }
    if key.pressed(KeyCode::KeyR) {
        let mut transform = transform_query.get_mut(build_data.root_ship_entity.unwrap()).unwrap();
        let mut orig_rotation = transform.rotation.to_euler(EulerRot::XYZ);
        orig_rotation.1 += 0.01;
        println!("rotation {:?}",orig_rotation);
        transform.rotation = Quat::from_euler(EulerRot::XYZ, orig_rotation.0, orig_rotation.1, orig_rotation.2);
    }
    // gizmo.cuboid(Transform::from_translation(Vec3::new(0.0,30.0,0.0)), Color::srgb_u8(0, 255, 0));
    // gizmo.cuboid(Transform::from_translation(Vec3::new(0.0,30.0,1.0)), Color::srgb_u8(0, 255, 0));
    //
    // let up_plane = Thing::Plane(Vec3::ZERO, *Dir3::Y, *Dir3::Z, *Dir3::X);
    // let vertex = Thing::Vertex(Vec3 { x: 0.0, y: 3.5, z: 0.0 });
    // let line1 = Thing::Line(
    //     Vec3 { x: 0.0, y: 5.0, z: 0.0 },
    //     Vec3 { x: 0.01, y: 10.0, z: 3.0 }
    // );
    // let line2 = Thing::Line(
    //     Vec3 { x: -0.01, y: -4.0, z: -0.1 },
    //     Vec3 { x: 0.0, y: -8.0, z: 0.0 }
    // );
    //
    //
    //
    // let line3 = Thing::Line(
    //     Vec3::new(1.0,10.0,-1.0),
    //     Vec3::new(-1.0,5.0,1.0)
    // );
    //
    // let line4 = Thing::Line(
    //     Vec3::new(-1.0,-10.0,-1.0),
    //     Vec3::new(1.0,-5.0,1.0)
    // );
    //
    // if let Thing::Line(start,end) = line3 {gizmo.line(start, end, Color::srgb_u8(255, 0, 0));}
    // if let Thing::Line(start,end) = line4 {gizmo.line(start, end, Color::srgb_u8(0, 0, 255));}
    // gizmo.cuboid(Transform::from_scale(Vec3::new(2.0,20.0,2.0)), Color::WHITE);



    //println!("totouchthing {:?}",to_touch_thing(&vertex, &up_plane, &Dir3::NEG_Y,&mut gizmo));
    //println!("totouchthing {:?}",to_touch_thing(&line1, &line2, &Dir3::NEG_Y,&mut gizmo));
    //println!("totouchthing {:?}",to_touch_thing(&line3, &line4, &Dir3::NEG_Y,&mut gizmo));

    // let cube_1 = Transform::from_translation(Vec3::new(10.0, 10.0, 10.0)).with_scale(Vec3::new(5.0,1.0,5.0));
    // for cube_thing in all_things(&cube_1) {
    //     println!("a thing for cube_1 is {:?}",cube_thing);
    // }


    // if scenes.is_changed() {
    //     for scene in scenes.iter() {
    //         for entity in scene.1.world.iter_entities() {
    //             println!("the thing is {:?}",entity.id());
    //         }
    //     }
    //
    // }

    //if key.just_pressed(KeyCode::KeyJ) {
    //    for mut pair in &mut hull_query {

    //        let mut mesh = Mesh::new(bevy::render::mesh::PrimitiveTopology::TriangleList,RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD);

    //        generate_adjustable_hull_mesh(
    //            &mut mesh,
    //            &pair.2
    //        );

    //        commands.entity(pair.0).insert((
    //            Mesh3d(meshes.add(mesh)),
    //            MeshMaterial3d(materials.add(pair.1.color))
    //        ));
    //    }
    //}
}







#[derive(Resource)]
struct InitData {
    file_path: Option<String>,
    data_paths: Option<LocalPaths>
}

#[derive(Resource)]
struct BuildData{
    root_ship_entity: Option<Entity>
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


    let mut light_transform = Transform::from_xyz(500.0, 500.0, 500.0);
    light_transform = light_transform.looking_at(Vec3 {x:0.0, y:0.0, z:0.0 }, Vec3::Y);
    // light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        light_transform,
    ));
    ambient_light.brightness = 2000.0;


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
            watch_for_changes_override: Some(false),
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
        // .insert_resource(InitData {
        //     file_path,
        //     data_paths: data_paths.clone()
        // })
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
        })
        .add_plugins((
            // WebAssetPlugin,
            default_plugins,
            // NavalSketchPlugin{
            //     init_paths: data_paths.unwrap(),
            // },
            WireframePlugin::default(),
            WaterPlugin,
            CameraMovementPlugin,
            MeshPickingPlugin,
            //OutlinePlugin,
        ))
        .add_systems(Startup, (setup).chain())
        .add_systems(Update, (temp_test_update))


        .run();

}
