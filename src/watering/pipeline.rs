use bevy::prelude::*;
use bevy::render::render_resource::{AddressMode, Extent3d, FilterMode, SamplerDescriptor, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension};
use bevy::render::renderer::RenderDevice;
use crate::watering::{DIRECTIONS, GRID_X_SIZE, GRID_Z_SIZE, PROFILE_BUFFER_LENGTH, RENDER_X_SIZE, RENDER_Z_SIZE, TOTAL_WAVE_NUMBERS};
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

    // ── Profile buffer ─────────────────────────────────────────────────────
    // 1-D R32Float texture (stored as 1 × PROFILE_SIZE).
    // Ψ(p) = integral_k  sqrt(S(k)) · cos(k·p) dk
    // Written by the profile compute shader, read by height reconstruction.
    pub profile_buf:      Texture,
    pub profile_buf_view: TextureView,
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
            width:                 GRID_X_SIZE as u32,
            height:                GRID_Z_SIZE as u32,
            // indexed by (k*DIRECTIONS)+direction
            depth_or_array_layers: (DIRECTIONS * TOTAL_WAVE_NUMBERS) as u32,
        },
        mip_level_count:     1,
        sample_count:        1,
        dimension:           TextureDimension::D2,
        format:              TextureFormat::Rg32Float,
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

    // ── Profile buffer ─────────────────────────────────────────────────────
    let profile_buf = render_device.create_texture(&TextureDescriptor {
        label:               Some("profile_buf"),
        size:                Extent3d { width: PROFILE_BUFFER_LENGTH as u32, height: 1, depth_or_array_layers: 1 },
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
        size:                Extent3d { width: RENDER_X_SIZE as u32, height: RENDER_Z_SIZE as u32, depth_or_array_layers: 1 },
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
        profile_buf,
        profile_buf_view,
        // height_field,
        // height_field_view,
        // height_field_sampler,
    });
}
