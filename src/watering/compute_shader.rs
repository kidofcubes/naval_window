use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel};
use bevy::render::render_resource::{encase, AddressMode, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer, BufferInitDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, DynamicUniformBuffer, Extent3d, FilterMode, PipelineCache, SamplerDescriptor, ShaderStages, StorageTextureAccess, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension};
use bevy::render::render_resource::binding_types::{sampler, texture_2d_array, texture_3d, texture_storage_2d, texture_storage_2d_array, uniform_buffer};
use bevy::render::render_resource::Sampler;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use crate::watering::{HeightFieldHandle, VoxelGridHandle, VoxelSettings, WaterDebugSettings, WaterSettings, WaterWaveGroupSpeeds, DIRECTIONS, GRID_X_SIZE, GRID_Z_SIZE, PROFILE_BUFFER_LENGTH, RENDER_X_SIZE, RENDER_Z_SIZE, TOTAL_WAVE_NUMBERS};
// use bevy::render::render_resource::encase::matrix::MatrixScalar;
// use bevy::render::render_resource::encase::matrix::AsMutMatrixParts;

// ─── GPU Resources ────────────────────────────────────────────────────────────

/// to be put in render world
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

    // ── Amplitude sampler ──────────────────────────────────────────────────
    // Linear filtering sampler for bilinear interpolation of amplitude textures.
    pub amplitude_sampler: Sampler,

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


#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct WaterNodeLabel;

#[derive(Default)]
pub struct WaterSimulationNode;

impl render_graph::Node for WaterSimulationNode {
    fn run(
        &self,
        _graph:  &mut RenderGraphContext,
        ctx:     &mut RenderContext,
        world:   &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipelines      = world.resource::<WaterPipelines>();
        let Some(bgs)      = world.get_resource::<WaterBindGroups>() else { return Ok(()) };

        // Fetch compiled pipelines — skip if any aren't ready yet.
        let (Some(inj_pl), Some(adv_pl), Some(diff_pl), Some(prof_pl), Some(hgt_pl)) = (
            pipeline_cache.get_compute_pipeline(pipelines.inject_id),
            pipeline_cache.get_compute_pipeline(pipelines.advection_id),
            pipeline_cache.get_compute_pipeline(pipelines.diffusion_id),
            pipeline_cache.get_compute_pipeline(pipelines.profile_id),
            pipeline_cache.get_compute_pipeline(pipelines.height_id),
        ) else {
            return Ok(());
        };

        // ── Workgroup counts ───────────────────────────────────────────────
        // Each shader uses @workgroup_size(64, 64, 1) over the grid / profile.
        let wg_grid    = (GRID_X_SIZE.div_ceil(32) as u32, GRID_Z_SIZE.div_ceil(32) as u32, DIRECTIONS as u32);
        let wg_profile = (PROFILE_BUFFER_LENGTH.div_ceil(128) as u32, 1, 1);
        let wg_height  = (RENDER_X_SIZE.div_ceil(32) as u32, RENDER_Z_SIZE.div_ceil(32) as u32, 1);

        let encoder = ctx.command_encoder();

        // ── Pass 0 : Debug Injection ───────────────────────────────────────
        // Inject amplitude at debug location (runs every frame, but shader
        // checks the inject flag and early-outs if not active).
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label:            Some("water_inject"),
                timestamp_writes: None,
            });
            pass.set_pipeline(inj_pl);
            pass.set_bind_group(0, &bgs.inject, &[]);
            pass.dispatch_workgroups(wg_grid.0, wg_grid.1, wg_grid.2);
        }

        // ── Pass 1 : Advection ─────────────────────────────────────────────
        // Semi-Lagrangian advection of A(x, y, θ_d).
        // Dispatched once per direction in the Z dimension of the workgroup.
        // Reads from amp_read, writes to amp_write.
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label:            Some("water_advection"),
                timestamp_writes: None,
            });
            pass.set_pipeline(adv_pl);
            pass.set_bind_group(0, &bgs.advection, &[]);
            pass.dispatch_workgroups(wg_grid.0, wg_grid.1, wg_grid.2);
        }

        // ── Pass 2 : Diffusion ────────────────────────────────────────────
        // Directional diffusion of A(x, y, θ_d) - spreads energy between adjacent directions.
        // Reads from amp_write, writes back to amp_read.
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label:            Some("water_diffusion"),
                timestamp_writes: None,
            });
            pass.set_pipeline(diff_pl);
            pass.set_bind_group(0, &bgs.diffusion, &[]);
            pass.dispatch_workgroups(wg_grid.0, wg_grid.1, wg_grid.2);
        }

        // ── Pass 3 : Profile buffer ────────────────────────────────────────
        // Build Ψ(p) from the Phillips spectrum.
        // Done after advection so height reconstruction uses an up-to-date profile.
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label:            Some("water_profile"),
                timestamp_writes: None,
            });
            pass.set_pipeline(prof_pl);
            pass.set_bind_group(0, &bgs.profile, &[]);
            pass.dispatch_workgroups(wg_profile.0, wg_profile.1, wg_profile.2);
        }

        // ── Pass 4 : Height reconstruction ─────────────────────────────────
        // η(x, y) = Σ_d  A(x, y, d) · Ψ(k̂_d · x_world + φ_d)
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label:            Some("water_height"),
                timestamp_writes: None,
            });
            pass.set_pipeline(hgt_pl);
            pass.set_bind_group(0, &bgs.height, &[]);
            pass.dispatch_workgroups(wg_height.0, wg_height.1, wg_height.2);
        }

        Ok(())
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
        usage:               TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
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

    // ── Amplitude sampler ─────────────────────────────────────────────────
    let amplitude_sampler = render_device.create_sampler(&SamplerDescriptor {
        label:           Some("amplitude_sampler"),
        address_mode_u:  AddressMode::ClampToEdge,
        address_mode_v:  AddressMode::ClampToEdge,
        mag_filter:      FilterMode::Linear,
        min_filter:      FilterMode::Linear,
        ..default()
    });

    // ── Profile buffer ─────────────────────────────────────────────────────
    let profile_buf = render_device.create_texture(&TextureDescriptor {
        label:               Some("profile_buf"),
        size:                Extent3d { width: PROFILE_BUFFER_LENGTH as u32, height: TOTAL_WAVE_NUMBERS as u32, depth_or_array_layers: 1 },
        mip_level_count:     1,
        sample_count:        1,
        dimension:           TextureDimension::D2,
        format:              TextureFormat::Rgba32Float,
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
        format:              TextureFormat::Rgba32Float,
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
        amplitude_sampler,
        profile_buf,
        profile_buf_view,
    });
}

// ─── Pipeline Resource ────────────────────────────────────────────────────────

/// Holds all compute pipelines and their shared bind group layouts.
#[derive(Resource)]
pub struct WaterPipelines {
    // ── Layouts ────────────────────────────────────────────────────────────
    pub inject_layout:    BindGroupLayoutDescriptor,
    pub advection_layout: BindGroupLayoutDescriptor,
    pub diffusion_layout: BindGroupLayoutDescriptor,
    pub profile_layout:   BindGroupLayoutDescriptor,
    pub height_layout:    BindGroupLayoutDescriptor,

    // ── Cached pipeline IDs ────────────────────────────────────────────────
    /// Pass 0 – debug amplitude injection (optional, runs when inject flag is set).
    pub inject_id:    CachedComputePipelineId,
    /// Pass 1 – semi-Lagrangian advection of A(x, y, θ).
    pub advection_id: CachedComputePipelineId,
    /// Pass 2 – directional diffusion of A(x, y, θ).
    pub diffusion_id: CachedComputePipelineId,
    /// Pass 3 – build the 1-D profile buffer Ψ(p) from the Phillips spectrum.
    pub profile_id:   CachedComputePipelineId,
    /// Pass 4 – reconstruct the visible height field η(x, y).
    pub height_id:    CachedComputePipelineId,
}

impl FromWorld for WaterPipelines {
    fn from_world(world: &mut World) -> Self {
        let device        = world.resource::<RenderDevice>();
        let asset_server  = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // ── Inject layout ─────────────────────────────────────────────────
        // Binding 0 : amplitude_out (Texture2DArray, ReadWrite storage)
        // Binding 1 : settings      (Uniform buffer)
        // Binding 2 : debug_settings (Uniform buffer)
        let inject_layout = BindGroupLayoutDescriptor::new(
            "inject_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d_array(
                        TextureFormat::Rg32Float,
                        StorageTextureAccess::ReadWrite,
                    ),
                    uniform_buffer::<WaterSettings>(false),
                    uniform_buffer::<WaterDebugSettings>(false),
                ),
            ),
        );

        // ── Advection layout ───────────────────────────────────────────────
        // Binding 0 : amplitude_in  (Texture2DArray, sampled)
        // Binding 1 : amplitude_sampler (Sampler)
        // Binding 2 : amplitude_out (Texture2DArray, WriteOnly storage)
        // Binding 3 : settings      (Uniform buffer)
        // Binding 4 : voxel_texture (Texture3D, Uint)
        // Binding 5 : voxel_settings (Uniform buffer)
        let advection_layout = BindGroupLayoutDescriptor::new(
            "advection_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_2d_array(bevy::render::render_resource::TextureSampleType::Float { filterable: true }),
                    sampler(bevy::render::render_resource::SamplerBindingType::Filtering),
                    texture_storage_2d_array(
                        TextureFormat::Rg32Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                    uniform_buffer::<WaterSettings>(false),
                    texture_3d(bevy::render::render_resource::TextureSampleType::Uint),
                    uniform_buffer::<VoxelSettings>(false),
                    uniform_buffer::<WaterWaveGroupSpeeds>(false),
                ),
            ),
        );

        // ── Diffusion layout ──────────────────────────────────────────────────
        // Binding 0 : amplitude_in  (Texture2DArray, sampled)
        // Binding 1 : amplitude_sampler (Sampler)
        // Binding 2 : amplitude_out (Texture2DArray, WriteOnly storage)
        // Binding 3 : settings      (Uniform buffer)
        let diffusion_layout = BindGroupLayoutDescriptor::new(
            "diffusion_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_2d_array(bevy::render::render_resource::TextureSampleType::Float { filterable: true }),
                    sampler(bevy::render::render_resource::SamplerBindingType::Filtering),
                    texture_storage_2d_array(
                        TextureFormat::Rg32Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                    uniform_buffer::<WaterSettings>(false),
                    uniform_buffer::<WaterWaveGroupSpeeds>(false),
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
                        TextureFormat::Rgba32Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                    uniform_buffer::<WaterSettings>(false),
                ),
            ),
        );

        // ── Height layout ──────────────────────────────────────────────────
        // Binding 0 : amplitude (Texture2DArray, sampled)
        // Binding 1 : amplitude_sampler (Sampler)
        // Binding 2 : profile   (Texture2D, ReadOnly storage)
        // Binding 3 : height out (Texture2D, WriteOnly storage)
        // Binding 4 : normal out (Texture2D, WriteOnly storage, rgba32float)
        // Binding 5 : settings  (Uniform)
        let height_layout = BindGroupLayoutDescriptor::new(
            "height_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_2d_array(bevy::render::render_resource::TextureSampleType::Float { filterable: true }),
                    sampler(bevy::render::render_resource::SamplerBindingType::Filtering),
                    texture_storage_2d(
                        TextureFormat::Rgba32Float,
                        StorageTextureAccess::ReadOnly,
                    ),
                    texture_storage_2d(
                        TextureFormat::Rgba32Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                    texture_storage_2d(
                        TextureFormat::Rgba32Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                    uniform_buffer::<WaterSettings>(false),
                ),
            ),
        );

        // ── Load shaders ───────────────────────────────────────────────────
        let inject_shader  = asset_server.load("water_inject.wgsl");
        let adv_shader     = asset_server.load("water_advection.wgsl");
        let diffusion_shader = asset_server.load("water_diffusion.wgsl");
        let profile_shader = asset_server.load("water_profile.wgsl");
        let height_shader  = asset_server.load("water_height.wgsl");

        // ── Queue pipelines ────────────────────────────────────────────────
        let inject_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                        Some("water_inject".into()),
            layout:                       vec![inject_layout.clone()],
            push_constant_ranges:         vec![],
            shader:                       inject_shader,
            shader_defs:                  vec![],
            entry_point:                  Some("inject_amplitude".into()),
            zero_initialize_workgroup_memory: false,
        });

        let advection_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                        Some("water_advection".into()),
            layout:                       vec![advection_layout.clone()],
            push_constant_ranges:         vec![],
            shader:                       adv_shader,
            shader_defs:                  vec![],
            entry_point:                  Some("advect".into()),
            zero_initialize_workgroup_memory: false,
        });

        let diffusion_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                        Some("water_diffusion".into()),
            layout:                       vec![diffusion_layout.clone()],
            push_constant_ranges:         vec![],
            shader:                       diffusion_shader,
            shader_defs:                  vec![],
            entry_point:                  Some("diffuse".into()),
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
            inject_layout,
            advection_layout,
            diffusion_layout,
            profile_layout,
            height_layout,
            inject_id,
            advection_id,
            diffusion_id,
            profile_id,
            height_id,
        }
    }
}

// ─── Bind Groups (rebuilt each frame because ping-pong flips) ─────────────────

#[derive(Resource)]
pub struct WaterBindGroups {
    pub inject:    BindGroup,
    pub advection: BindGroup,
    pub diffusion: BindGroup,
    pub profile:   BindGroup,
    pub height:    BindGroup,
    /// Settings uniform buffer (updated via write_buffer each frame).
    pub settings_buf: Buffer,
    /// Debug settings uniform buffer.
    pub debug_settings_buf: Buffer,
    /// Voxel settings uniform buffer.
    pub voxel_settings_buf: Buffer,
    pub water_wave_group_speeds_buf: Buffer,
}

pub fn queue_bind_groups(
    mut commands:  Commands,
    device:        Res<RenderDevice>,
    pipelines:     Res<WaterPipelines>,
    gpu:           Option<ResMut<WaterGpuResources>>,
    settings:      Res<WaterSettings>,
    debug_settings: Res<WaterDebugSettings>,
    voxel_settings: Res<VoxelSettings>,
    voxel_handle:  Res<VoxelGridHandle>,
    water_wave_group_speeds:  Res<WaterWaveGroupSpeeds>,
    existing_bgs:  Option<ResMut<WaterBindGroups>>,
    hf_handle:     Option<ResMut<HeightFieldHandle>>,
    render_images: Res<RenderAssets<GpuImage>>,
    queue:         Res<bevy::render::renderer::RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(mut gpu) = gpu else { return };

    let Some(hf_handle) = hf_handle else { return };
    // Get the GpuImage Bevy uploaded from your Image asset
    let Some(offsets_gpu_image) = render_images.get(&hf_handle.offsets_handle) else { return };
    let Some(normal_gpu_image) = render_images.get(&hf_handle.normals_handle) else { return };

    // Get the voxel grid texture
    let Some(voxel_handle) = &voxel_handle.0 else { return };
    let Some(voxel_gpu_image) = render_images.get(voxel_handle) else { return };
    let voxel_texture_view = &voxel_gpu_image.texture_view;

    // This texture_view is the exact same texture the material will sample.
    // Use it as the compute shader's write target.
    let height_write_view = &offsets_gpu_image.texture_view;
    let normal_write_view = &normal_gpu_image.texture_view;

    // ── Create or update the uniform buffers ────────────────────────────────
    let (settings_buf, debug_settings_buf, voxel_settings_buf, water_wave_group_speeds_buf) = if let Some(mut bgs) = existing_bgs {
        // Write updated settings into the existing buffers.
        let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
        buf.write(&*settings).expect("ShaderType write failed");
        queue.write_buffer(&bgs.settings_buf, 0, buf.into_inner().as_slice());

        let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
        buf.write(&*debug_settings).expect("ShaderType write failed");
        queue.write_buffer(&bgs.debug_settings_buf, 0, buf.into_inner().as_slice());

        let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
        buf.write(&*voxel_settings).expect("ShaderType write failed");
        queue.write_buffer(&bgs.voxel_settings_buf, 0, buf.into_inner().as_slice());

        // no water_wave_group_speeds_buf because it doesn't need to be updated

        // Flip ping-pong *after* updating, then rebuild bind groups
        // (they reference the new read/write views).
        gpu.flip = !gpu.flip;
        (bgs.settings_buf.clone(), bgs.debug_settings_buf.clone(), bgs.voxel_settings_buf.clone(), bgs.water_wave_group_speeds_buf.clone())
    } else {
        // First frame – create buffers from scratch.
        let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
        buf.write(&*settings).expect("ShaderType write failed");
        let settings_buf = device.create_buffer_with_data(&BufferInitDescriptor {
            label:    Some("water_settings_buf"),
            contents: &buf.into_inner(),
            usage:    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
        buf.write(&*debug_settings).expect("ShaderType write failed");
        let debug_settings_buf = device.create_buffer_with_data(&BufferInitDescriptor {
            label:    Some("water_debug_settings_buf"),
            contents: &buf.into_inner(),
            usage:    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
        buf.write(&*voxel_settings).expect("ShaderType write failed");
        let voxel_settings_buf = device.create_buffer_with_data(&BufferInitDescriptor {
            label:    Some("water_voxel_settings_buf"),
            contents: &buf.into_inner(),
            usage:    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let mut buf = encase::DynamicUniformBuffer::new_with_alignment(Vec::<u8>::new(), 32);
        buf.write(&*water_wave_group_speeds).expect("ShaderType write failed");
        let water_wave_group_speeds_buf = device.create_buffer_with_data(&BufferInitDescriptor {
            label:    Some("water_wave_group_speeds_buf"),
            contents: &buf.into_inner(),
            usage:    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        (settings_buf, debug_settings_buf, voxel_settings_buf, water_wave_group_speeds_buf)
    };

    let (amp_read, amp_write) = gpu.amp_views();

    // Inject writes to the *write* texture (before advection reads from it next frame)
    let inject = device.create_bind_group(
        "inject_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.inject_layout),
        &BindGroupEntries::sequential((
            amp_read,
            settings_buf.as_entire_binding(),
            debug_settings_buf.as_entire_binding(),
        )),
    );

    // Advection reads from amp_read, writes to amp_write
    let advection = device.create_bind_group(
        "advection_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.advection_layout),
        &BindGroupEntries::sequential((
            amp_read,
            &gpu.amplitude_sampler,
            amp_write,
            settings_buf.as_entire_binding(),
            voxel_texture_view,
            voxel_settings_buf.as_entire_binding(),
            water_wave_group_speeds_buf.as_entire_binding(),
        )),
    );

    // Diffusion reads from amp_write (advection output), writes back to amp_read
    let diffusion = device.create_bind_group(
        "diffusion_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.diffusion_layout),
        &BindGroupEntries::sequential((
            amp_write,
            &gpu.amplitude_sampler,
            amp_read,
            settings_buf.as_entire_binding(),
            water_wave_group_speeds_buf.as_entire_binding(),
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

    // Height reconstruction reads the diffused amplitude (amp_read after diffusion writes to it).
    let height = device.create_bind_group(
        "height_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.height_layout),
        &BindGroupEntries::sequential((
            amp_read,
            &gpu.amplitude_sampler,
            &gpu.profile_buf_view,
            height_write_view,
            normal_write_view,
            settings_buf.as_entire_binding(),
        )),
    );

    commands.insert_resource(WaterBindGroups {
        inject,
        advection,
        diffusion,
        profile,
        height,
        settings_buf,
        debug_settings_buf,
        voxel_settings_buf,
        water_wave_group_speeds_buf,
    });
}