use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::*;

/// CPU-based wave simulation for ship wakes
#[derive(Resource)]
pub struct WakeSimulation {
    // Double buffer for wave heights
    pub current_heights: Vec<f32>,
    previous_heights: Vec<f32>,

    // Simulation parameters
    pub resolution: usize,
    pub plane_size: f32,
    // pub plane_offset: Vec2,
    pub wave_speed: f32,
    pub damping: f32,

    // GPU texture
    pub texture_handle: Handle<Image>,
    needs_update: bool,
}

impl WakeSimulation {
    pub fn new(
        images: &mut Assets<Image>,
        resolution: usize,
        world_size: f32,
    ) -> Self {
        let pixel_count = resolution * resolution;
        let current_heights = vec![0.0; pixel_count];
        let previous_heights = vec![0.0; pixel_count];

        // Create the GPU texture
        let mut image = Image::new_fill(
            Extent3d {
                width: resolution as u32,
                height: resolution as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &vec![0u8; pixel_count * 4], // R32Float = 4 bytes per pixel
            TextureFormat::R32Float,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );

        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

        let texture_handle = images.add(image);

        Self {
            current_heights,
            previous_heights,
            resolution,
            plane_size: world_size,
            // plane_offset: Vec2::ZERO,
            wave_speed: 5.0,
            damping: 0.995,
            texture_handle,
            needs_update: true,
        }
    }

    /// Update the wave simulation
    pub fn step(&mut self, delta_time: f32) {
        let res = self.resolution;
        let grid_spacing = self.plane_size / res as f32;
        let dt_sq = delta_time * delta_time;
        let c_sq = self.wave_speed * self.wave_speed;

        // Create temporary buffer for new heights
        let mut new_heights = vec![0.0; res * res];

        for y in 1..res-1 {
            for x in 1..res-1 {
                let idx = y * res + x;

                let current = self.current_heights[idx];
                let previous = self.previous_heights[idx];

                // Sample neighbors for Laplacian
                let north = self.current_heights[(y + 1) * res + x];
                let south = self.current_heights[(y - 1) * res + x];
                let east = self.current_heights[y * res + (x + 1)];
                let west = self.current_heights[y * res + (x - 1)];

                // Compute Laplacian (discrete approximation of ∇²u)
                let laplacian = (north + south + east + west - 4.0 * current)
                    / (grid_spacing * grid_spacing);

                // Wave equation: d²u/dt² = c² * ∇²u
                let new_height = 2.0 * current - previous + c_sq * dt_sq * laplacian;

                // Apply damping
                new_heights[idx] = new_height * self.damping;
            }
        }

        // Handle boundaries (simple clamping to zero at edges)
        for i in 0..res {
            new_heights[i] = 0.0; // Top edge
            new_heights[(res - 1) * res + i] = 0.0; // Bottom edge
            new_heights[i * res] = 0.0; // Left edge
            new_heights[i * res + (res - 1)] = 0.0; // Right edge
        }

        // Swap buffers
        std::mem::swap(&mut self.previous_heights, &mut self.current_heights);
        self.current_heights = new_heights;

        self.needs_update = true;
    }

    /// Add an impulse at a world position (pos is 0..(plane_size-1))
    pub fn add_impulse(&mut self, pos: Vec2, strength: f32, radius: f32) {
        let res = self.resolution;

        for y in 0..res {
            for x in 0..res {
                // Convert pixel to world position
                let dist = vec2(((x as f32)/res as f32) * self.plane_size, ((y as f32)/res as f32) * self.plane_size).distance(pos);

                // Gaussian impulse
                if dist < radius * 3.0 {
                    let impulse = strength * (-dist * dist / (radius * radius)).exp();
                    let idx = y * res + x;
                    self.current_heights[idx] += impulse;
                }
            }
        }

        self.needs_update = true;
    }

    // /// Convert pixel coordinates to world position
    // fn pixel_to_world(&self, x: usize, y: usize) -> Vec2 {
    //     let uv = Vec2::new(x as f32 / self.resolution as f32, y as f32 / self.resolution as f32);
    //     self.plane_offset + (uv - Vec2::splat(0.5)) * self.plane_size
    // }

    // /// Convert world position to pixel coordinates
    // pub fn world_to_pixel(&self, world_pos: Vec2) -> Option<(usize, usize)> {
    //     let half_size = self.plane_size * 0.5;
    //     let relative = world_pos - self.plane_offset;
    //     let uv = (relative + Vec2::splat(half_size)) / self.plane_size;
    //
    //     if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
    //         return None;
    //     }
    //
    //     let x = (uv.x * self.resolution as f32) as usize;
    //     let y = (uv.y * self.resolution as f32) as usize;
    //     Some((x.min(self.resolution - 1), y.min(self.resolution - 1)))
    // }

    /// Upload CPU data to GPU texture
    pub fn update_texture(&mut self, images: &mut Assets<Image>) {
        if !self.needs_update {
            return;
        }

        let Some(image) = images.get_mut(&self.texture_handle) else {
            info!("wtf its not some whats going on");
            return;
        };

        // Convert f32 heights to bytes
        let bytes: Vec<u8> = self.current_heights
            .iter()
            .flat_map(|&h| h.to_le_bytes())
            .collect();

        image.data = Some(bytes);

        self.needs_update = false;
    }
}