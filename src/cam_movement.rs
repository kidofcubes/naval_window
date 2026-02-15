use bevy::{input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll}, pbr::ScreenSpaceAmbientOcclusion, prelude::*};
use std::f32::consts::FRAC_PI_2;
use bevy::picking::hover::HoverMap;

/// A vector representing the player's input, accumulated over all frames that ran
/// since the last time the physics simulation was advanced.
#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
pub struct AccumulatedInput(Vec3);

/// A vector representing the player's velocity in the physics simulation.
#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
pub struct Velocity(Vec3);

/// The actual position of the player in the physics simulation.
/// This is separate from the `Transform`, which is merely a visual representation.
///
/// If you want to make sure that this component is always initialized
/// with the same value as the `Transform`'s translation, you can
/// use a [component lifecycle hook](https://docs.rs/bevy/0.14.0/bevy/ecs/component/struct.ComponentHooks.html)
#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
pub struct PhysicalTranslation(Vec3);

/// The value [`PhysicalTranslation`] had in the last fixed timestep.
/// Used for interpolation in the `interpolate_rendered_transform` system.
#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
pub struct PreviousPhysicalTranslation(Vec3);

#[derive(Debug, Component, Clone, Copy, PartialEq, Default)]
pub struct EditorCamera;

pub struct CameraMovementPlugin;

impl Plugin for CameraMovementPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Startup, spawn_player)
            .add_systems(FixedUpdate, advance_physics)
            .add_systems(
                // The `RunFixedMainLoop` schedule allows us to schedule systems to run before and after the fixed timestep loop.
                RunFixedMainLoop,
                (
                    // The physics simulation needs to know the player's input, so we run this before the fixed timestep loop.
                    // Note that if we ran it in `Update`, it would be too late, as the physics simulation would already have been advanced.
                    // If we ran this in `FixedUpdate`, it would sometimes not register player input, as that schedule may run zero times per frame.
                    handle_input.in_set(RunFixedMainLoopSystems::BeforeFixedMainLoop),
                    // The player's visual representation needs to be updated after the physics simulation has been advanced.
                    // This could be run in `Update`, but if we run it here instead, the systems in `Update`
                    // will be working with the `Transform` that will actually be shown on screen.
                    interpolate_rendered_transform.in_set(RunFixedMainLoopSystems::AfterFixedMainLoop),
                ),
            )
            .add_systems(Update,(move_player,grab_mouse))
        ;
    }
}

/// Spawn the player sprite and a 2D camera.
pub fn spawn_player(mut commands: Commands) {
    let camera = Camera3d{
            
            ..default()
        };
    commands.spawn((
        camera,
        Msaa::Off,
        Name::new("Player"),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        AccumulatedInput::default(),
        Velocity::default(),
        PhysicalTranslation::default(),
        PreviousPhysicalTranslation::default(),
        ScreenSpaceAmbientOcclusion::default(),
        EditorCamera,
        // GizmoCamera,
    ));
}

/// Spawn a bit of UI text to explain how to move the player.
pub fn spawn_text(mut commands: Commands) {
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        })
        .with_child((
            Text::new("Move the player with WASD"),
            TextFont {
                font_size: 25.0,
                ..default()
            },
        ));
}

/// Handle keyboard input and accumulate it in the `AccumulatedInput` component.
///
/// There are many strategies for how to handle all the input that happened since the last fixed timestep.
/// This is a very simple one: we just accumulate the input and average it out by normalizing it.
pub fn handle_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut AccumulatedInput, &mut Velocity, &mut Transform)>,
) {
    /// Since Bevy's default 2D camera setup is scaled such that
    /// one unit is one pixel, you can think of this as
    /// "How many pixels per second should the player move?"
    const SPEED: f32 = 100.0;
    for (input, mut velocity, transform) in query.iter_mut() {
        // if keyboard_input.pressed(KeyCode::KeyW) {
        //     input.z -= 1.0;
        // }
        // if keyboard_input.pressed(KeyCode::KeyS) {
        //     input.z += 1.0;
        // }
        // if keyboard_input.pressed(KeyCode::KeyA) {
        //     input.x -= 1.0;
        // }
        // if keyboard_input.pressed(KeyCode::KeyD) {
        //     input.x += 1.0;
        // }
        // if keyboard_input.pressed(KeyCode::Space) {
        //     input.y += 1.0;
        // }
        // if keyboard_input.pressed(KeyCode::ShiftLeft) {
        //     input.y -= 1.0;
        // }

        // Need to normalize and scale because otherwise
        // diagonal movement would be faster than horizontal or vertical movement.
        // This effectively averages the accumulated input.
        velocity.0 = transform.rotation.mul_vec3(input.normalize_or_zero() * SPEED);
    }
}

/// Advance the physics simulation by one fixed timestep. This may run zero or multiple times per frame.
///
/// Note that since this runs in `FixedUpdate`, `Res<Time>` would be `Res<Time<Fixed>>` automatically.
/// We are being explicit here for clarity.
pub fn advance_physics(
    fixed_time: Res<Time<Fixed>>,
    mut query: Query<(
        &mut PhysicalTranslation,
        &mut PreviousPhysicalTranslation,
        &mut AccumulatedInput,
        &Velocity,
    )>,
) {
    for (
        mut current_physical_translation,
        mut previous_physical_translation,
        mut input,
        velocity,
    ) in query.iter_mut()
    {
        previous_physical_translation.0 = current_physical_translation.0;
        current_physical_translation.0 += velocity.0 * fixed_time.delta_secs();

        // Reset the input accumulator, as we are currently consuming all input that happened since the last fixed timestep.
        input.0 = Vec3::ZERO;
    }
}

pub fn interpolate_rendered_transform(
    fixed_time: Res<Time<Fixed>>,
    query: Query<(
        &mut Transform,
        &PhysicalTranslation,
        &PreviousPhysicalTranslation,
    )>,
) {
    // for (mut transform, current_physical_translation, previous_physical_translation) in
    //     query.iter_mut()
    // {
    //     let previous = previous_physical_translation.0;
    //     let current = current_physical_translation.0;
    //     // The overstep fraction is a value between 0 and 1 that tells us how far we are between two fixed timesteps.
    //     let alpha = fixed_time.overstep_fraction();
    //
    //     let rendered_translation = previous.lerp(current, alpha);
    //     transform.translation = rendered_translation;
    // }
}

pub fn move_player(
    accumulated_mouse_motion: Res<AccumulatedMouseMotion>,
    accumulated_mouse_scroll: Res<AccumulatedMouseScroll>,
    hover_map: Res<HoverMap>,
    ui_nodes: Query<&Node>,
    windows: Query<&mut Window>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut player: Query<&mut Transform, With<EditorCamera>>,
) {
    let window = windows.single();
    // if window.cursor_options.grab_mode != CursorGrabMode::Locked {
    //     return;
    // }

    let Ok(mut transform) = player.single_mut() else {
        return;
    };

    let delta = accumulated_mouse_motion.delta;

    if mouse.pressed(MouseButton::Right) {
        // Note that we are not multiplying by delta_time here.
        // The reason is that for mouse movement, we already get the full movement that happened since the last frame.
        // This means that if we multiply by delta_time, we will get a smaller rotation than intended by the user.
        // This situation is reversed when reading e.g. analog input from a gamepad however, where the same rules
        // as for keyboard input apply. Such an input should be multiplied by delta_time to get the intended rotation
        // independent of the framerate.
        let delta_yaw = -delta.x * 0.0025;
        let delta_pitch = -delta.y * 0.0025;

        let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
        let yaw = yaw + delta_yaw;

        // If the pitch was ±¹⁄₂ π, the camera would look straight up or down.
        // When the user wants to move the camera back to the horizon, which way should the camera face?
        // The camera has no way of knowing what direction was "forward" before landing in that extreme position,
        // so the direction picked will for all intents and purposes be arbitrary.
        // Another issue is that for mathematical reasons, the yaw will effectively be flipped when the pitch is at the extremes.
        // To not run into these issues, we clamp the pitch to a safe range.
        const PITCH_LIMIT: f32 = FRAC_PI_2 - 0.01;
        let pitch = (pitch + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);

        transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }
    

    for (_pointer, pointer_map) in hover_map.iter() {
        for (entity, _hit) in pointer_map.iter() {
            if ui_nodes.contains(*entity) {
                return; //skip scroll movement when in ui
            }
        }
    }

    let mut translation = Vec3::ZERO;

    if mouse.pressed(MouseButton::Middle) {
        translation += ((transform.left()*delta.x)+(transform.up()*delta.y))*0.01;
    }
    
    
    
    translation+= if mouse.pressed(MouseButton::Forward)||mouse.pressed(MouseButton::Back) {transform.up()} else {transform.forward()}*accumulated_mouse_scroll.delta.y;
    translation+= transform.right()*accumulated_mouse_scroll.delta.x;
    transform.translation+=translation;
}

pub fn grab_mouse(
    mut windows: Query<&mut Window>,
    mouse: Res<ButtonInput<MouseButton>>,
    key: Res<ButtonInput<KeyCode>>,
) {
    let window = windows.single_mut();

    // if mouse.just_pressed(MouseButton::Middle) {
    //     window.cursor_options.visible = false;
    //     window.cursor_options.grab_mode = CursorGrabMode::Locked;
    // }
    // if mouse.just_released(MouseButton::Middle) {
    //     window.cursor_options.visible = true;
    //     window.cursor_options.grab_mode = CursorGrabMode::None;
    // }

    // if mouse.just_pressed(MouseButton::Left) {
    //     window.cursor_options.visible = false;
    //     window.cursor_options.grab_mode = CursorGrabMode::Locked;
    //     let new_pos = Some(
    //             DVec2 { x: window.physical_width() as f64/2.0, y: window.physical_height() as f64/2.0}
    //         );
    //     window.set_physical_cursor_position(new_pos);
    // }
    //
    // if key.just_pressed(KeyCode::Escape) {
    //     window.cursor_options.visible = true;
    //     window.cursor_options.grab_mode = CursorGrabMode::None;
    //     let new_pos = Some(
    //             DVec2 { x: window.physical_width() as f64/2.0, y: window.physical_height() as f64/2.0}
    //         );
    //     window.set_physical_cursor_position(new_pos);
    // }
}



