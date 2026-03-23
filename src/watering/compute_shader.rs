use std::collections::HashMap;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel};
use bevy::render::render_resource::{encase, AddressMode, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer, BufferInitDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, DynamicUniformBuffer, Extent3d, FilterMode, IntoBindGroupLayoutEntryBuilderArray, PipelineCache, Sampler, SamplerDescriptor, ShaderStages, StorageTextureAccess, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension};
use bevy::render::render_resource::binding_types::{sampler, texture_2d_array, texture_3d, texture_storage_2d, texture_storage_2d_array, uniform_buffer};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use crate::watering::{VoxelGridHandle, VoxelSettings, WaterDebugSettings, WaterSettings, WaterWaveGroupSpeeds, DIRECTIONS, GRID_X_SIZE, GRID_Z_SIZE, PROFILE_BUFFER_LENGTH, RENDER_X_SIZE, RENDER_Z_SIZE, TOTAL_ZETA_NUMBERS, WaterSimulationHandles};

// ─── GPU Resources ────────────────────────────────────────────────────────────

/// to be put in render world
#[derive(Resource)]
pub struct WaterGpuResources {
    pub clamp_linear_sampler: Sampler,
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

        // ── Workgroup counts ───────────────────────────────────────────────
        // Each shader uses @workgroup_size(64, 64, 1) over the grid / profile.
        let wg_grid    = (GRID_X_SIZE.div_ceil(32) as u32, GRID_Z_SIZE.div_ceil(32) as u32, DIRECTIONS as u32);
        let wg_profile = (PROFILE_BUFFER_LENGTH.div_ceil(128) as u32, 1, 1);

        let encoder = ctx.command_encoder();

        let to_run: Vec<Option<(String, &ComputePipeline, &BindGroup, (u32,u32,u32))>> =
            vec![
                ("inject", wg_grid),
                ("water_advection", wg_grid),
                ("water_diffusion", wg_grid),
                ("water_profile", wg_profile),
            ].into_iter().map( |(name, wg)| {
                let pipeline = pipelines.passes.get(name).and_then(|x|  pipeline_cache.get_compute_pipeline(x.1));
                pipeline.map(|pipeline| (
                    name.to_owned().to_owned(),
                    pipeline,
                    wg
                ))
                .and_then(|(name, pipeline, wg)| {
                    bgs.bind_groups.get(&name).map(|bg| (name, pipeline, bg, wg))
                })
            }).collect::<Vec<_>>();

        for opt in to_run.iter() {
            if(opt.is_none()){
                return Ok(());
            }
        }
        let to_run = to_run.into_iter().map(|x| x.unwrap()).collect::<Vec<_>>();

        for (name, pipeline, bg, wg) in to_run {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label:            Some(&name),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, wg.2);
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
    // ── Amplitude sampler ─────────────────────────────────────────────────
    let clamp_linear_sampler = render_device.create_sampler(&SamplerDescriptor {
        label:           Some("amplitude_sampler"),
        address_mode_u:  AddressMode::ClampToEdge,
        address_mode_v:  AddressMode::ClampToEdge,
        mag_filter:      FilterMode::Linear,
        min_filter:      FilterMode::Linear,
        ..default()
    });
    commands.insert_resource(WaterGpuResources {
        clamp_linear_sampler,
    });
}

// ─── Pipeline Resource ────────────────────────────────────────────────────────

/// Creates a `(BindGroupLayoutDescriptor, CachedComputePipelineId)` pair.
///
/// `$label` is used as the pipeline label and `"$label.wgsl"` as the shader path.
/// Entry point is always `"main"`.
/// `$entries` is the tuple passed to `BindGroupLayoutEntries::sequential`.
macro_rules! compute_pipeline {
    ($passes_map:expr, $label:expr, $asset_server:expr, $pipeline_cache:expr, $entries:expr) => {{
        let layout = BindGroupLayoutDescriptor::new(
            concat!($label, "_layout"),
            &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, $entries),
        );
        let id = $pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                           Some(($label).into()),
            layout:                          vec![layout.clone()],
            push_constant_ranges:            vec![],
            shader:                          $asset_server.load(concat!($label, ".wgsl")),
            shader_defs:                     vec![],
            entry_point:                     Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });
        $passes_map.insert(layout, id);
    }};
}

/// Holds all compute pipelines and their shared bind group layouts.
#[derive(Resource)]
pub struct WaterPipelines {
    pub passes: HashMap<String, (BindGroupLayoutDescriptor, CachedComputePipelineId)>,
}

impl FromWorld for WaterPipelines {
    fn from_world(world: &mut World) -> Self {
        let mut passes = HashMap::new();

        // let device        = world.resource::<RenderDevice>();
        let asset_server  = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        compute_pipeline!(passes, "water_inject", asset_server, pipeline_cache, (
            texture_storage_2d_array(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
            uniform_buffer::<WaterSettings>(false),
            uniform_buffer::<WaterDebugSettings>(false),
        ));

        compute_pipeline!(passes, "water_advection", asset_server, pipeline_cache, (
            texture_2d_array(bevy::render::render_resource::TextureSampleType::Float { filterable: true }),
            sampler(bevy::render::render_resource::SamplerBindingType::Filtering),
            texture_storage_2d_array(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
            uniform_buffer::<WaterSettings>(false),
            texture_3d(bevy::render::render_resource::TextureSampleType::Uint),
            uniform_buffer::<VoxelSettings>(false),
            uniform_buffer::<WaterWaveGroupSpeeds>(false),
            texture_3d(bevy::render::render_resource::TextureSampleType::Float { filterable: false }),
        ));

        compute_pipeline!(passes, "water_diffusion", asset_server, pipeline_cache, (
            texture_2d_array(bevy::render::render_resource::TextureSampleType::Float { filterable: true }),
            sampler(bevy::render::render_resource::SamplerBindingType::Filtering),
            texture_storage_2d_array(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
            uniform_buffer::<WaterSettings>(false),
            uniform_buffer::<WaterWaveGroupSpeeds>(false),
        ));

        compute_pipeline!(passes, "water_profile", asset_server, pipeline_cache, (
            texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            uniform_buffer::<WaterSettings>(false),
        ));

        Self {
            passes: HashMap::new(),
        }
    }
}

// ─── Bind Groups (rebuilt each frame because ping-pong flips) ─────────────────

#[derive(Resource)]
pub struct WaterBindGroups {
    bind_groups: HashMap<String, BindGroup>,
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
    // hf_handle:     Option<ResMut<HeightFieldHandle>>,
    sim_handle:     Option<ResMut<WaterSimulationHandles>>,
    render_images: Res<RenderAssets<GpuImage>>,
    queue:         Res<bevy::render::renderer::RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(mut gpu) = gpu else { return };

    // let Some(hf_handle) = hf_handle else { return };
    let Some(sim_handle) = sim_handle else { return };
    // Get the GpuImage Bevy uploaded from your Image asset
    // let Some(offsets_gpu_image) = render_images.get(&hf_handle.offsets_handle) else { return };
    // let Some(normal_gpu_image) = render_images.get(&hf_handle.normals_handle) else { return };

    let Some(amp_a) = render_images.get(&sim_handle.amplitude_texture_a) else { return };
    let Some(amp_b) = render_images.get(&sim_handle.amplitude_texture_b) else { return };
    let Some(profile_buffer) = render_images.get(&sim_handle.profile_buffer) else { return };

    // Get the voxel grid texture
    let Some(voxel_handle_inner) = &voxel_handle.voxel else { return };
    let Some(voxel_gpu_image) = render_images.get(voxel_handle_inner) else { return };
    let voxel_texture_view = &voxel_gpu_image.texture_view;

    // Get the SDF texture
    let Some(sdf_handle) = &voxel_handle.sdf else { return };
    let Some(sdf_gpu_image) = render_images.get(sdf_handle) else { return };
    let sdf_texture_view = &sdf_gpu_image.texture_view;

    // This texture_view is the exact same texture the material will sample.
    // Use it as the compute shader's write target.
    // let height_write_view = &offsets_gpu_image.texture_view;
    // let normal_write_view = &normal_gpu_image.texture_view;


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

    // let (amp_read, amp_write) = gpu.amp_views();

    let (amp_read, amp_write) = if sim_handle.flipped {
        (&amp_b.texture_view, &amp_a.texture_view)
    } else {
        (&amp_a.texture_view, &amp_b.texture_view)
    };

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
            &gpu.clamp_linear_sampler,
            amp_write,
            settings_buf.as_entire_binding(),
            voxel_texture_view,
            voxel_settings_buf.as_entire_binding(),
            water_wave_group_speeds_buf.as_entire_binding(),
            sdf_texture_view,
        )),
    );

    // Diffusion reads from amp_write (advection output), writes back to amp_read
    let diffusion = device.create_bind_group(
        "diffusion_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.diffusion_layout),
        &BindGroupEntries::sequential((
            amp_write,
            &gpu.clamp_linear_sampler,
            amp_read,
            settings_buf.as_entire_binding(),
            water_wave_group_speeds_buf.as_entire_binding(),
        )),
    );

    let profile = device.create_bind_group(
        "profile_bg",
        &pipeline_cache.get_bind_group_layout(&pipelines.profile_layout),
        &BindGroupEntries::sequential((
            // &gpu.profile_buf_view,
            &profile_buffer.texture_view,
            settings_buf.as_entire_binding(),
        )),
    );

    commands.insert_resource(WaterBindGroups {
        inject,
        advection,
        diffusion,
        profile,
        settings_buf,
        debug_settings_buf,
        voxel_settings_buf,
        water_wave_group_speeds_buf,
    });
}