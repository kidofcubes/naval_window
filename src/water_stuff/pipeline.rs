use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::RenderDevice,
    },
};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{texture_storage_2d, texture_storage_2d_array, uniform_buffer};
use bevy::render::texture::GpuImage;
use super::{WaterSettings, resources::WaterGpuResources, HeightFieldHandle};

// ─── Pipeline Resource ────────────────────────────────────────────────────────

/// Holds all three compute pipelines and their shared bind group layouts.
#[derive(Resource)]
pub struct WaterPipelines {
    // ── Layouts ────────────────────────────────────────────────────────────
    pub advection_layout: BindGroupLayoutDescriptor,
    pub profile_layout:   BindGroupLayoutDescriptor,
    pub height_layout:    BindGroupLayoutDescriptor,

    // ── Cached pipeline IDs ────────────────────────────────────────────────
    /// Pass 1 – semi-Lagrangian advection of A(x, y, θ).
    pub advection_id: CachedComputePipelineId,
    /// Pass 2 – build the 1-D profile buffer Ψ(p) from the Phillips spectrum.
    pub profile_id:   CachedComputePipelineId,
    /// Pass 3 – reconstruct the visible height field η(x, y).
    pub height_id:    CachedComputePipelineId,
}

impl FromWorld for WaterPipelines {
    fn from_world(world: &mut World) -> Self {
        let device        = world.resource::<RenderDevice>();
        let asset_server  = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // ── Advection layout ───────────────────────────────────────────────
        // Binding 0 : amplitude_in  (Texture2DArray, ReadOnly storage)
        // Binding 1 : amplitude_out (Texture2DArray, WriteOnly storage)
        // Binding 2 : obstacle_mask (Texture2D, ReadOnly storage)
        // Binding 3 : settings      (Uniform buffer)
        // let advection_layout = device.create_bind_group_layout(
        let advection_layout = BindGroupLayoutDescriptor::new(
            "advection_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d_array(
                        TextureFormat::R32Float,
                        StorageTextureAccess::ReadOnly,
                    ),
                    texture_storage_2d_array(
                        TextureFormat::R32Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                    texture_storage_2d(
                        TextureFormat::R8Unorm,
                        StorageTextureAccess::ReadOnly,
                    ),
                    uniform_buffer::<WaterSettings>(false),
                ),
            ),
        );

        // ── Profile layout ─────────────────────────────────────────────────
        // Binding 0 : profile_out (Texture2D 1×N, WriteOnly)
        // Binding 1 : settings    (Uniform)
        // let profile_layout = device.create_bind_group_layout(
        let profile_layout = BindGroupLayoutDescriptor::new(
            "profile_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(
                        TextureFormat::R32Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                    uniform_buffer::<WaterSettings>(false),
                ),
            ),
        );

        // ── Height layout ──────────────────────────────────────────────────
        // Binding 0 : amplitude (Texture2DArray, ReadOnly)
        // Binding 1 : profile   (Texture2D, ReadOnly)
        // Binding 2 : height out (Texture2D, WriteOnly)
        // Binding 3 : settings  (Uniform)
        // let height_layout = device.create_bind_group_layout(
        let height_layout = BindGroupLayoutDescriptor::new(
            "height_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d_array(
                        TextureFormat::R32Float,
                        StorageTextureAccess::ReadOnly,
                    ),
                    texture_storage_2d(
                        TextureFormat::R32Float,
                        StorageTextureAccess::ReadOnly,
                    ),
                    texture_storage_2d(
                        TextureFormat::R32Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                    uniform_buffer::<WaterSettings>(false),
                ),
            ),
        );

        // ── Load shaders ───────────────────────────────────────────────────
        let adv_shader     = asset_server.load("water_advection.wgsl");
        let profile_shader = asset_server.load("water_profile.wgsl");
        let height_shader  = asset_server.load("water_height.wgsl");

        // ── Queue pipelines ────────────────────────────────────────────────
        let advection_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                        Some("water_advection".into()),
            layout:                       vec![advection_layout.clone()],
            push_constant_ranges:         vec![],
            shader:                       adv_shader,
            shader_defs:                  vec![],
            entry_point:                  Some("advect".into()),
            zero_initialize_workgroup_memory: false,
        });

        let profile_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                        Some("water_profile".into()),
            layout:                       vec![profile_layout.clone()],
            push_constant_ranges:         vec![],
            shader:                       profile_shader,
            shader_defs:                  vec![],
            entry_point:                  Some("build_profile".into()),
            zero_initialize_workgroup_memory: false,
        });

        let height_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                        Some("water_height".into()),
            layout:                       vec![height_layout.clone()],
            push_constant_ranges:         vec![],
            shader:                       height_shader,
            shader_defs:                  vec![],
            entry_point:                  Some("reconstruct_height".into()),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            advection_layout,
            profile_layout,
            height_layout,
            advection_id,
            profile_id,
            height_id,
        }
    }
}

// ─── Bind Groups (rebuilt each frame because ping-pong flips) ─────────────────

#[derive(Resource)]
pub struct WaterBindGroups {
    pub advection: BindGroup,
    pub profile:   BindGroup,
    pub height:    BindGroup,
    /// Settings uniform buffer (updated via write_buffer each frame).
    pub settings_buf: Buffer,
}

pub fn queue_bind_groups(
    mut commands:  Commands,
    device:        Res<RenderDevice>,
    pipelines:     Res<WaterPipelines>,
    gpu:           Option<ResMut<WaterGpuResources>>,
    settings:      Res<WaterSettings>,
    existing_bgs:  Option<ResMut<WaterBindGroups>>,
    hf_handle:  Option<ResMut<HeightFieldHandle>>,
    render_images: Res<RenderAssets<GpuImage>>,
    queue:         Res<bevy::render::renderer::RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(mut gpu) = gpu else { return };

    let Some(hf_handle) = hf_handle else { return };
    // Get the GpuImage Bevy uploaded from your Image asset
    let Some(gpu_image) = render_images.get(&hf_handle.0) else { return };

    // This texture_view is the exact same texture the material will sample.
    // Use it as the compute shader's write target.
    let height_write_view = &gpu_image.texture_view;

    // ── Create or update the settings uniform buffer ───────────────────────
    let settings_buf = if let Some(mut bgs) = existing_bgs {
        // Write updated settings into the existing buffer.
        let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
        buf.write(&*settings).expect("ShaderType write failed");
        queue.write_buffer(&bgs.settings_buf, 0, buf.into_inner().as_slice());

        // Flip ping-pong *after* updating, then rebuild bind groups
        // (they reference the new read/write views).
        gpu.flip = !gpu.flip;
        bgs.settings_buf.clone() // borrow workaround – see below
    } else {
        // First frame – create buffer from scratch.
        let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
        buf.write(&*settings).expect("ShaderType write failed");
        let bytes = buf.into_inner();
        device.create_buffer_with_data(&BufferInitDescriptor {
            label:    Some("water_settings_buf"),
            contents: &bytes,
            usage:    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        })
    };

    let (amp_read, amp_write) = gpu.amp_views();

    let advection = device.create_bind_group(
        "advection_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.advection_layout),
        &BindGroupEntries::sequential((
            amp_read,
            amp_write,
            &gpu.obstacle_mask_view,
            settings_buf.as_entire_binding(),
        )),
    );

    let profile = device.create_bind_group(
        "profile_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.profile_layout),
        &BindGroupEntries::sequential((
            &gpu.profile_buf_view,
            settings_buf.as_entire_binding(),
        )),
    );

    // Height reconstruction reads the *newly written* amplitude (amp_write).
    let height = device.create_bind_group(
        "height_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.height_layout),
        &BindGroupEntries::sequential((
            amp_write,
            &gpu.profile_buf_view,
            height_write_view,
            settings_buf.as_entire_binding(),
        )),
    );

    commands.insert_resource(WaterBindGroups {
        advection,
        profile,
        height,
        settings_buf,
    });
}
