mod pipeline;

use std::f32::consts::TAU;
use std::time::Instant;
use bevy::app::{App, Plugin, Startup, Update};
use bevy::prelude::*;
use bevy::prelude::ops::atan2;
use bevy::render::renderer::RenderDevice;
use bevy::render::settings::WgpuFeatures;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct Watering;
impl Plugin for Watering{
    fn build(&self, app: &mut App) {
        // app.add_plugins(MaterialPlugin::<CustomMaterial>{
        //     ..default()
        // });
        app.insert_resource(WateringStuff::default());
        app.insert_resource(WateringStuffTest {
            scale: 5.0,
            radius: 5.0,
            spawn_center: IVec2::new(GRID_X_SIZE as i32/2,GRID_Z_SIZE as i32/2),
        });
        // app.add_systems(Startup, );
        app.add_systems(Update, (update,debug_update).chain());
    }
}
const GRID_X_SIZE: usize = 256;
const GRID_Z_SIZE: usize = 256;
const DIRECTIONS: usize = 16;
const TOTAL_WAVE_NUMBERS: usize = 1;
const WAVE_NUMBERS: [f32; TOTAL_WAVE_NUMBERS] = [1.0];


const RENDER_X_SIZE: usize = 256;
const RENDER_Z_SIZE: usize = 256;

#[derive(Resource)]
struct WateringStuffTest {
    scale: f32,
    radius: f32,
    spawn_center: IVec2,
}

#[derive(Resource)]
struct WateringStuff {
    /// [x][z][theta][samples] .x is real component .y is imaginary
    pub sim_grid: Box<[[[[Vec2; TOTAL_WAVE_NUMBERS]; DIRECTIONS]; GRID_Z_SIZE]; GRID_X_SIZE]>,
    /// [x][z][theta][samples] .x is real component .y is imaginary
    pub new_sim_grid: Box<[[[[Vec2; TOTAL_WAVE_NUMBERS]; DIRECTIONS]; GRID_Z_SIZE]; GRID_X_SIZE]>,
    pub height_map: Box<[[f32; RENDER_Z_SIZE]; RENDER_X_SIZE]>
}
impl Default for WateringStuff {
    fn default() -> Self {
        // let sim_grid = vec![[[[0.0; TOTAL_WAVE_NUMBERS]; DIRECTIONS]; GRID_Z_SIZE]; GRID_X_SIZE];
        // let new_sim_grid = vec![[[[0.0; TOTAL_WAVE_NUMBERS]; DIRECTIONS]; GRID_Z_SIZE]; GRID_X_SIZE];
        // let height_map = vec![[0.0; RENDER_Z_SIZE]; RENDER_X_SIZE];
        Self {
            // sim_grid: [[[[0.0; TOTAL_WAVE_NUMBERS]; DIRECTIONS]; GRID_Z_SIZE]; GRID_X_SIZE],
            // sim_grid: sim_grid.try_into().unwrap(),
            sim_grid: Box::new([[[[vec2(0.0,0.0); TOTAL_WAVE_NUMBERS]; DIRECTIONS]; GRID_Z_SIZE]; GRID_X_SIZE]),
            // new_sim_grid: [[[[0.0; TOTAL_WAVE_NUMBERS]; DIRECTIONS]; GRID_Z_SIZE]; GRID_X_SIZE],
            // new_sim_grid: new_sim_grid.try_into().unwrap(),
            new_sim_grid: Box::new([[[[vec2(0.0,0.0); TOTAL_WAVE_NUMBERS]; DIRECTIONS]; GRID_Z_SIZE]; GRID_X_SIZE]),
            // height_map: [[0.0; RENDER_Z_SIZE]; RENDER_X_SIZE],
            // height_map: height_map.try_into().unwrap(),
            height_map: Box::new([[0.0; RENDER_Z_SIZE]; RENDER_X_SIZE]),
        }
    }
 }


trait AmplitudeGrid {
    fn bilinear_sample(&self, x: f32, z: f32, theta_index: usize, wave_index: usize) -> Vec2;
}

impl AmplitudeGrid for [[[[Vec2; TOTAL_WAVE_NUMBERS]; DIRECTIONS]; GRID_Z_SIZE]; GRID_X_SIZE] {
    fn bilinear_sample(&self, x: f32, z: f32, theta_index: usize, wave_index: usize) -> Vec2{
        // if(x < 0.0 || x >= GRID_X_SIZE as f32 || z < 0.0 || z >= GRID_Z_SIZE as f32){
        if(x < 1.0 || x >= (GRID_X_SIZE-1) as f32 || z < 1.0 || z >= (GRID_Z_SIZE-1) as f32){
            //outside grid, assume zero
            return vec2(0.0,0.0);
        }
        let x_0 = x.floor() as usize;
        let x_1 = x_0+1;
        let z_0 = z.floor() as usize;
        let z_1 = z_0+1;

        let x_mix = (x - x_0 as f32);
        let z_mix = (z - z_0 as f32);

        // println!("x: {}, z: {}, theta_index: {}, wave_index: {}", x, z, theta_index, wave_index);
        let value = (
                (self[x_0][z_0][theta_index][wave_index] * (1.0-x_mix) * (1.0-z_mix)) +
                (self[x_1][z_0][theta_index][wave_index] * (    x_mix) * (1.0-z_mix)) +
                (self[x_1][z_1][theta_index][wave_index] * (    x_mix) * (    z_mix)) +
                (self[x_0][z_1][theta_index][wave_index] * (1.0-x_mix) * (    z_mix))
        );

        return value;
    }
}

//smaller wave number = bigger wave length
//wave number is how many peaks per 2pi meters in this case

/// meters/(second^2)
const GRAVITY: f32 = 9.81;


const PROFILE_BUFFER_LENGTH: usize = 2048;
const TIME_MULT: f32 = 3.0;

fn update(
    mut watering_stuff: ResMut<WateringStuff>,
    time: Res<Time>
) {
    let mut start_instant: Instant;
    start_instant =  Instant::now();
    for wave_index in 0..TOTAL_WAVE_NUMBERS {

        // how many waves per 2pi meters
        let wave_number = WAVE_NUMBERS[wave_index];

        // how fast each individual trough/crest moves
        let phase_speed = f32::sqrt(GRAVITY / wave_number);
        // how far the energy of the wave moves per second
        let group_speed = phase_speed / 2.0;


        for theta_index in 0..DIRECTIONS {
            let theta = index_to_radians(theta_index);

            for x in 0..GRID_X_SIZE {
                for z in 0..GRID_Z_SIZE {
                    let dx = ( f32::cos(theta) * (group_speed * time.delta_secs() * TIME_MULT));
                    let dz = ( f32::sin(theta) * (group_speed * time.delta_secs() * TIME_MULT));
                    let prev_x = (x as f32) - dx;
                    let prev_z = (z as f32) - dz;
                    let prev = watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32+0i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index);
                    let phase = atan2(prev.y,prev.x);
                    let normalized = prev.clone().normalize();
                    // let magnitude = prev.length();
                    // prev_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32-2i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 0.10;
                    // prev_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32-1i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 0.25;
                    // prev_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32+0i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 0.30;
                    // prev_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32+1i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 0.25;
                    // prev_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32+2i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 0.10;


                    // new_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32-1i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 0.15;
                    // new_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32+0i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 0.70;
                    // new_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32+1i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 0.15;

                    let mut new_magnitude = 0.0;

                    new_magnitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32-1i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index).length() * 0.15;
                    new_magnitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32+0i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index).length() * 0.70;
                    new_magnitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32+1i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index).length() * 0.15;

                    // new_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x, prev_z, (theta_index as i32+0i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 1.00;

                    // new_amplitude *= 0.5;
                    //
                    // new_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x - dx, prev_z - dx, (theta_index as i32+0i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 0.25;
                    // new_amplitude += watering_stuff.sim_grid.bilinear_sample(prev_x + dx, prev_z + dx, (theta_index as i32+0i32).rem_euclid(DIRECTIONS as i32) as usize, wave_index) * 0.25;

                    let mut new_amplitude = vec2(0.0,0.0);
                    new_amplitude = prev.clone().normalize_or(vec2(1.0,0.0))*new_magnitude;
                    // new_amplitude = vec2(f32::cos(phase), f32::sin(phase)) * new_magnitude;
                    watering_stuff.new_sim_grid[x][z][theta_index][wave_index] = new_amplitude;

                    //todo do the other random stuff to make it look better later

                }
            }
        }
    }
    // println!("elapsed {}ms for advection", start_instant.elapsed().as_millis());
    let watering_stuff_mut = watering_stuff.as_mut();
    std::mem::swap(&mut watering_stuff_mut.sim_grid, &mut watering_stuff_mut.new_sim_grid);


    let mut r = StdRng::seed_from_u64(0);

    //plug in position offset*wave_number and the rest is precomputed
    let mut profile_buffer: [[f32; PROFILE_BUFFER_LENGTH]; TOTAL_WAVE_NUMBERS] = [[0.0; PROFILE_BUFFER_LENGTH]; TOTAL_WAVE_NUMBERS];

    for wave_index in 0..TOTAL_WAVE_NUMBERS {
        let wave_number = WAVE_NUMBERS[wave_index];
        let angular_frequency = f32::sqrt(GRAVITY * wave_number);
        let phase_coordinate = ((time.elapsed_secs() * TIME_MULT) * angular_frequency);

        for p in 0..PROFILE_BUFFER_LENGTH {
            let offset = (p as f32 / PROFILE_BUFFER_LENGTH as f32) * TAU;
            // let phase_coordinate = (wave_number*position_offset) - ((time.elapsed_secs() * TIME_MULT) * angular_frequency);

            profile_buffer[wave_index][p] += f32::cos(offset-phase_coordinate);
        }
    }
    // let mut offset = r.random_range(0.0..TAU)/4.0;


    start_instant = Instant::now();
    for x in 0..RENDER_X_SIZE{
        for z in 0..RENDER_Z_SIZE{
            watering_stuff.height_map[x][z] = 0.0;
            let real_x = (x as f32 / RENDER_X_SIZE as f32) * GRID_X_SIZE as f32;
            let real_z = (z as f32 / RENDER_Z_SIZE as f32) * GRID_Z_SIZE as f32;

            for theta_index in 0..DIRECTIONS {
                let theta = index_to_radians(theta_index);
                for wave_index in 0..TOTAL_WAVE_NUMBERS {
                    let wave_number = WAVE_NUMBERS[wave_index];
                    // let amplitude = watering_stuff.sim_grid[x][z][theta_index][wave_index];
                    let amplitude = watering_stuff.sim_grid.bilinear_sample(real_x,real_z,theta_index, wave_index);

                    let position_offset = vec2(real_x,real_z).dot(vec2(f32::cos(theta),f32::sin(theta)));
                    let angular_frequency = f32::sqrt(GRAVITY * wave_number);

                    let phase_coordinate = (wave_number*position_offset) - ((time.elapsed_secs() * TIME_MULT) * angular_frequency);
                    // let profile_buffer_coordinate = (wave_number*position_offset);
                    let profile_buffer_index = ((((wave_number*position_offset)/TAU) * PROFILE_BUFFER_LENGTH as f32) as i32).rem_euclid(PROFILE_BUFFER_LENGTH as i32) as usize;
                    if (x==0&&z==0) || (x==RENDER_X_SIZE-1&&z==RENDER_Z_SIZE-1) {
                        // println!("phase_coordinate {}", (((phase_coordinate/TAU) * PROFILE_BUFFER_LENGTH as f32) as i32));
                    }
                    // watering_stuff.height_map[x][z] += amplitude * profile_buffer[theta_index][(((phase_coordinate/TAU) * PROFILE_BUFFER_LENGTH as f32) as i32).rem_euclid(PROFILE_BUFFER_LENGTH as i32) as usize];
                    // watering_stuff.height_map[x][z] += amplitude * (phase_coordinate).cos();
                    // watering_stuff.height_map[x][z] += (amplitude.x * (phase_coordinate).cos()) - (amplitude.y * (phase_coordinate).sin());
                    watering_stuff.height_map[x][z] +=
                        (amplitude.x * profile_buffer[wave_index][profile_buffer_index]) -
                            (amplitude.y * profile_buffer[wave_index][(profile_buffer_index+((3*PROFILE_BUFFER_LENGTH)/4))%PROFILE_BUFFER_LENGTH]);
                    // watering_stuff.height_map[x][z] += (amplitude.x * (phase_coordinate).cos());
                }
            }
        }
    }
    // println!("elapsed {}ms for into heightmapping", start_instant.elapsed().as_millis());
}

fn debug_update(
    mut watering_stuff: ResMut<WateringStuff>,
    mut test_resource: ResMut<WateringStuffTest>,
    key: Res<ButtonInput<KeyCode>>,
    render_device: Res<RenderDevice>,
    mut gizmo: Gizmos,
    time: Res<Time>
) {
    // let limits = render_device.limits();
    // println!("max 1d {}",limits.max_texture_dimension_1d);
    // println!("max 2d {}",limits.max_texture_dimension_2d);
    // println!("max 3d {}",limits.max_texture_dimension_3d);
    // println!("max array layers {}",limits.max_texture_array_layers);
    // println!("has float filterable {}",render_device.features().features_wgpu.contains(WgpuFeatures::FLOAT32_FILTERABLE.features_wgpu));

    for x in 0..RENDER_X_SIZE {
        for z in 0..RENDER_Z_SIZE {
            let real_x = (x as f32 / RENDER_X_SIZE as f32) * GRID_X_SIZE as f32;
            let real_z = (z as f32 / RENDER_Z_SIZE as f32) * GRID_Z_SIZE as f32;
            gizmo.line(
                vec3(real_x-(RENDER_X_SIZE as f32/2.0),0.0,real_z-(RENDER_Z_SIZE as f32/2.0)),
                vec3(real_x-(RENDER_X_SIZE as f32/2.0),watering_stuff.height_map[x][z],real_z-(RENDER_Z_SIZE as f32/2.0)), Color::srgb((x as f32/RENDER_X_SIZE as f32),(z as f32/RENDER_Z_SIZE as f32), 0.5))
        }
    }
    if(key.pressed(KeyCode::KeyH)){
        let radius = test_resource.radius; //grid radius
        for x_offset in -(radius.ceil() as i32)..=(radius.ceil() as i32) {
            for z_offset in -(radius.ceil() as i32)..=(radius.ceil() as i32) {
                let real_x = test_resource.spawn_center.x + x_offset;
                let real_z = test_resource.spawn_center.y + z_offset;

                if(vec2(real_x as f32,real_z as f32).distance(vec2(test_resource.spawn_center.x as f32,test_resource.spawn_center.y as f32)) > radius){
                    continue;
                }

                for wave_index in 0..TOTAL_WAVE_NUMBERS {
                    let wave_number = WAVE_NUMBERS[wave_index];
                    let angular_frequency = f32::sqrt(GRAVITY * wave_number);

                    for theta_index in 0..DIRECTIONS {
                        let mut scale = test_resource.scale;
                        let theta = index_to_radians(theta_index);
                        let mut position_offset = vec2(real_x as f32, real_z as f32).dot(vec2(f32::cos(theta), f32::sin(theta)));

                        let mut phase_coordinate = (wave_number*position_offset) - ((time.elapsed_secs() * TIME_MULT) * angular_frequency);
                        // let mut phase_coordinate = - ((time.elapsed_secs() * TIME_MULT) * angular_frequency);
                        // let phase_coordinate = wave_number * position_offset;
                        // watering_stuff.sim_grid[GRID_X_SIZE / 2][GRID_Z_SIZE / 2][theta_index][wave_index] = vec2(10.0, TAU * (test_resource.complex_component / 16.0));
                        // watering_stuff.sim_grid[GRID_X_SIZE / 2][GRID_Z_SIZE / 2][theta_index][wave_index] = vec2(scale * f32::cos(-phase_coordinate), scale * f32::sin(-phase_coordinate));

                        // this only counteracts time, how to counteract position?
                        watering_stuff.sim_grid[real_x as usize][real_z as usize][theta_index][wave_index] =
                            vec2(scale * f32::cos(-phase_coordinate), scale * f32::sin(-phase_coordinate));
                    }
                }
            }
        }
    }

    if(key.pressed(KeyCode::ArrowUp)){
        test_resource.scale += 1.0;
        println!("scale {}", test_resource.scale);
    }

    if(key.pressed(KeyCode::ArrowDown)){
        test_resource.scale -= 1.0;
        println!("scale {}", test_resource.scale);
    }

    if(key.pressed(KeyCode::ArrowRight)){
        test_resource.radius += 1.0;
        println!("radius {}", test_resource.radius);
    }

    if(key.pressed(KeyCode::ArrowLeft)){
        test_resource.radius -= 1.0;
        println!("raidus {}", test_resource.radius);
    }

    if(key.pressed(KeyCode::KeyA)) {
        test_resource.spawn_center.x -= 1;
        println!("spawn_center {}",test_resource.spawn_center);
    }

    if(key.pressed(KeyCode::KeyD)) {
        test_resource.spawn_center.x += 1;
        println!("spawn_center {}",test_resource.spawn_center);
    }

    if(key.pressed(KeyCode::KeyS)) {
        test_resource.spawn_center.y -= 1;
        println!("spawn_center {}",test_resource.spawn_center);
    }

    if(key.pressed(KeyCode::KeyW)) {
        test_resource.spawn_center.y += 1;
        println!("spawn_center {}",test_resource.spawn_center);
    }
}


fn index_to_radians(i: usize) -> f32{
    return i as f32 * (std::f32::consts::TAU / DIRECTIONS as f32);
}