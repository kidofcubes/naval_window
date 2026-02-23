use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
};

use super::{DOMAIN_SIZE, GRID_SIZE, NUM_DIRECTIONS, PROFILE_SIZE, WaterSettings};

// ─── CPU-side Obstacle Mask ───────────────────────────────────────────────────

/// Updated each frame from your ship's voxel grid. Call `rasterize_voxels`
/// or manually call `set(x, y, solid)` from your ship system.
///
/// Coordinate space: (0,0) = domain corner at (-DOMAIN_SIZE/2, -DOMAIN_SIZE/2).
/// x → world +X,  y → world +Z.
#[derive(Resource, Default)]
pub struct ObstacleMask {
    /// Raw R8 data: 255 = solid, 0 = water. Row-major, GRID_SIZE wide.
    pub data: Vec<u8>,
    /// Set to true when `data` has changed so the GPU copy is triggered.
    pub dirty: bool,
}

impl ObstacleMask {
    pub fn new() -> Self {
        Self {
            data: vec![0u8; (GRID_SIZE * GRID_SIZE) as usize],
            dirty: true,
        }
    }

    /// Mark a single grid cell solid or clear.
    pub fn set(&mut self, x: u32, y: u32, solid: bool) {
        if x < GRID_SIZE && y < GRID_SIZE {
            self.data[(y * GRID_SIZE + x) as usize] = if solid { 255 } else { 0 };
            self.dirty = true;
        }
    }

    /// Clear all obstacles.
    pub fn clear(&mut self) {
        self.data.fill(0);
        self.dirty = true;
    }

    /// Convert a world-space XZ position to grid coordinates.
    pub fn world_to_grid(world_x: f32, world_z: f32) -> (u32, u32) {
        let half = DOMAIN_SIZE * 0.5;
        let gx = ((world_x + half) / DOMAIN_SIZE * GRID_SIZE as f32) as u32;
        let gy = ((world_z + half) / DOMAIN_SIZE * GRID_SIZE as f32) as u32;
        (gx.min(GRID_SIZE - 1), gy.min(GRID_SIZE - 1))
    }

    /// Rasterise a world-space AABB (at the waterline) into the mask.
    /// Call this each frame after clearing, passing the ship's cross-section.
    pub fn rasterize_aabb(&mut self, min_x: f32, min_z: f32, max_x: f32, max_z: f32) {
        let (gx0, gy0) = Self::world_to_grid(min_x, min_z);
        let (gx1, gy1) = Self::world_to_grid(max_x, max_z);
        for gy in gy0..=gy1.min(GRID_SIZE - 1) {
            for gx in gx0..=gx1.min(GRID_SIZE - 1) {
                self.data[(gy * GRID_SIZE + gx) as usize] = 255;
            }
        }
        self.dirty = true;
    }

    /// Rasterise an arbitrary set of world-space (x, z) voxel centroids
    /// projected onto the waterline.  Pass the slice of voxels that
    /// intersect the current waterline from your voxel grid.
    pub fn rasterize_voxels(&mut self, voxel_world_positions: &[(f32, f32)]) {
        for &(wx, wz) in voxel_world_positions {
            let (gx, gy) = Self::world_to_grid(wx, wz);
            self.data[(gy * GRID_SIZE + gx) as usize] = 255;
        }
        self.dirty = true;
    }
}

// ─── GPU Resources ────────────────────────────────────────────────────────────

/// All GPU textures owned by the water simulation.
#[derive(Resource)]
pub struct WaterGpuResources {
    // ── Amplitude ping-pong ────────────────────────────────────────────────
    // Two Texture2DArrays, each with NUM_DIRECTIONS layers.
    // Each layer stores A(x, y) for one propagation direction θ_d.
    // We alternate read ↔ write each frame to avoid read-write hazards.
    pub amplitude_a:      Texture,
    pub amplitude_a_view: TextureView,
    pub amplitude_b:      Texture,
    pub amplitude_b_view: TextureView,
    /// false → A is read, B is write.  true → B is read, A is write.
    pub flip: bool,

    // ── Obstacle mask ──────────────────────────────────────────────────────
    // R8Unorm 2D texture.  255 = solid hull, 0 = open water.
    // Updated from CPU via write_texture each frame when dirty.
    pub obstacle_mask:      Texture,
    pub obstacle_mask_view: TextureView,

    // ── Profile buffer ─────────────────────────────────────────────────────
    // 1-D R32Float texture (stored as 1 × PROFILE_SIZE).
    // Ψ(p) = integral_k  sqrt(S(k)) · cos(k·p) dk
    // Written by the profile compute shader, read by height reconstruction.
    pub profile_buf:      Texture,
    pub profile_buf_view: TextureView,

    // ── Reconstructed height field ─────────────────────────────────────────
    // R32Float 2D texture.  Readable by the vertex shader + CPU readback
    // for buoyancy calculations.
    pub height_field:         Texture,
    pub height_field_view:    TextureView,
    pub height_field_sampler: Sampler,
}

impl WaterGpuResources {
    /// Return (read_view, write_view) for the current frame.
    pub fn amp_views(&self) -> (&TextureView, &TextureView) {
        if self.flip {
            (&self.amplitude_b_view, &self.amplitude_a_view)
        } else {
            (&self.amplitude_a_view, &self.amplitude_b_view)
        }
    }
}

pub fn prepare_gpu_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    existing: Option<Res<WaterGpuResources>>,
) {
    // Only create once.
    if existing.is_some() {
        return;
    }

    // ── Amplitude arrays ───────────────────────────────────────────────────
    let amp_desc = TextureDescriptor {
        label:               None,
        size:                Extent3d {
            width:                 GRID_SIZE,
            height:                GRID_SIZE,
            depth_or_array_layers: NUM_DIRECTIONS,
        },
        mip_level_count:     1,
        sample_count:        1,
        dimension:           TextureDimension::D2,
        format:              TextureFormat::R32Float,
        usage:               TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
        view_formats:        &[],
    };
    let array_view_desc = TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    };

    let amplitude_a      = render_device.create_texture(&TextureDescriptor { label: Some("amp_a"), ..amp_desc });
    let amplitude_a_view = amplitude_a.create_view(&array_view_desc);
    let amplitude_b      = render_device.create_texture(&TextureDescriptor { label: Some("amp_b"), ..amp_desc });
    let amplitude_b_view = amplitude_b.create_view(&array_view_desc);

    // ── Obstacle mask ──────────────────────────────────────────────────────
    let obstacle_mask = render_device.create_texture(&TextureDescriptor {
        label:               Some("obstacle_mask"),
        size:                Extent3d { width: GRID_SIZE, height: GRID_SIZE, depth_or_array_layers: 1 },
        mip_level_count:     1,
        sample_count:        1,
        dimension:           TextureDimension::D2,
        format:              TextureFormat::R8Unorm,
        usage:               TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
        view_formats:        &[],
    });
    let obstacle_mask_view = obstacle_mask.create_view(&TextureViewDescriptor::default());

    // ── Profile buffer ─────────────────────────────────────────────────────
    let profile_buf = render_device.create_texture(&TextureDescriptor {
        label:               Some("profile_buf"),
        size:                Extent3d { width: PROFILE_SIZE, height: 1, depth_or_array_layers: 1 },
        mip_level_count:     1,
        sample_count:        1,
        dimension:           TextureDimension::D2,
        format:              TextureFormat::R32Float,
        usage:               TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats:        &[],
    });
    let profile_buf_view = profile_buf.create_view(&TextureViewDescriptor::default());

    // ── Height field ───────────────────────────────────────────────────────
    let height_field = render_device.create_texture(&TextureDescriptor {
        label:               Some("height_field"),
        size:                Extent3d { width: GRID_SIZE, height: GRID_SIZE, depth_or_array_layers: 1 },
        mip_level_count:     1,
        sample_count:        1,
        dimension:           TextureDimension::D2,
        format:              TextureFormat::R32Float,
        // TEXTURE_BINDING so the vertex shader can sample it via a sampler.
        // COPY_SRC so you can readback for buoyancy if needed.
        usage:               TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats:        &[],
    });
    let height_field_view = height_field.create_view(&TextureViewDescriptor::default());
    let height_field_sampler = render_device.create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        mag_filter:     FilterMode::Linear,
        min_filter:     FilterMode::Linear,
        ..default()
    });

    commands.insert_resource(WaterGpuResources {
        amplitude_a,
        amplitude_a_view,
        amplitude_b,
        amplitude_b_view,
        flip: false,
        obstacle_mask,
        obstacle_mask_view,
        profile_buf,
        profile_buf_view,
        height_field,
        height_field_view,
        height_field_sampler,
    });
}

/// Upload obstacle mask data to GPU when it's been marked dirty.
pub fn upload_obstacle_mask(
    mut mask: ResMut<ObstacleMask>,
    gpu: Option<ResMut<WaterGpuResources>>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(mut gpu), true) = (gpu, mask.dirty) else { return };

    render_queue.write_texture(
        TexelCopyTextureInfo {
            texture:   &gpu.obstacle_mask,
            mip_level: 0,
            origin:    Origin3d::ZERO,
            aspect:    TextureAspect::All,
        },
        &mask.data,
        TexelCopyBufferLayout {
            offset:         0,
            bytes_per_row:  Some(GRID_SIZE),
            rows_per_image: None,
        },
        Extent3d { width: GRID_SIZE, height: GRID_SIZE, depth_or_array_layers: 1 },
    );

    mask.dirty = false;
}
