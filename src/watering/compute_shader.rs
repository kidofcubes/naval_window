use std::collections::HashMap;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{encase, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer, BufferInitDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, Sampler, ShaderStages, StorageTextureAccess, TextureFormat};
use bevy::render::render_resource::binding_types::{sampler, texture_2d_array, texture_3d, texture_storage_2d, texture_storage_2d_array, uniform_buffer};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use strum::AsRefStr;
use crate::watering::{VoxelSettings, WaterDebugSettings, WaterPingPong, WaterSettings, GpuTexture, GpuTextures, WaterWaveGroupSpeeds};
use crate::watering::shader_utils::{create_clamp_linear_sampler, encode_uniform, grid_workgroups, profile_workgroups};

// ─── Pass / Buffer Enums ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, AsRefStr)]
#[strum(serialize_all = "snake_case")]
pub enum ComputePasses {
    WaterInject,
    WaterAdvection,
    WaterDiffusion,
    WaterProfile,
}

impl ComputePasses {
    pub fn wgsl_path(&self) -> String {
        match self {
            ComputePasses::WaterInject | ComputePasses::WaterAdvection | ComputePasses::WaterDiffusion | ComputePasses::WaterProfile => {
                format!("realwaves/{}.wgsl", self.as_ref())
            }
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, AsRefStr)]
#[strum(serialize_all = "snake_case")]
pub enum Buffers {
    Settings,
    DebugSettings,
    VoxelSettings,
    WaterWaveGroupSpeeds,
}

// ─── GPU Resources ────────────────────────────────────────────────────────────

/// to be put in render world
#[derive(Resource)]
pub struct WaterGpuResources {
    pub clamp_linear_sampler: Sampler,
}

pub fn render_pass(
    world: &World,
    // view: ViewQuery<(&ExtractedCamera, &ViewTarget)>,
    mut ctx: RenderContext,
) -> Result<()> {
    let pipeline_cache = world.resource::<PipelineCache>();
    let pipelines      = world.resource::<WaterPipelines>();
    let Some(bgs)      = world.get_resource::<WaterBindGroups>() else { return Ok(()) };

    // ── Workgroup counts ───────────────────────────────────────────────
    let wg_grid    = grid_workgroups();
    let wg_profile = profile_workgroups();

    let passes_to_run = [
        (ComputePasses::WaterInject, wg_grid),
        (ComputePasses::WaterAdvection, wg_grid),
        (ComputePasses::WaterDiffusion, wg_grid),
        (ComputePasses::WaterProfile, wg_profile),
    ];

    // Resolve all pipelines + bind groups; bail if any aren't ready.
    let mut resolved = Vec::with_capacity(passes_to_run.len());
    for (pass, wg) in &passes_to_run {
        let Some((_, pipeline_id)) = pipelines.passes.get(pass) else { return Ok(()) };
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(*pipeline_id) else { return Ok(()) };
        let Some(bg) = bgs.bind_groups.get(pass) else { return Ok(()) };
        resolved.push((pass.as_ref(), pipeline, bg, *wg));
    }

    let encoder = ctx.command_encoder();
    for (label, pipeline, bg, wg) in resolved {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label:            Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(wg.0, wg.1, wg.2);
    }

    Ok(())
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
    commands.insert_resource(WaterGpuResources {
        clamp_linear_sampler: create_clamp_linear_sampler(&render_device),
    });
}

// ─── Pipeline Resource ────────────────────────────────────────────────────────

/// Creates a `(BindGroupLayoutDescriptor, CachedComputePipelineId)` pair.
///
/// `$label` is used as the pipeline label and `"$label.wgsl"` as the shader path.
/// Entry point is always `"main"`.
/// `$entries` is the tuple passed to `BindGroupLayoutEntries::sequential`.
macro_rules! compute_pipeline {
    ($passes_map:expr, $pass:expr, $asset_server:expr, $pipeline_cache:expr, $entries:expr) => {{
        let pass: ComputePasses = $pass;
        let label: &'static str = Box::leak(pass.as_ref().to_owned().into_boxed_str());
        let layout_label: &'static str = Box::leak(format!("{}_layout", label).into_boxed_str());
        let layout = BindGroupLayoutDescriptor::new(
            layout_label,
            &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, $entries),
        );
        let id = $pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                           Some(label.into()),
            layout:                          vec![layout.clone()],
            shader:                          $asset_server.load(pass.wgsl_path()),
            shader_defs:                     vec![],
            entry_point:                     Some("main".into()),
            zero_initialize_workgroup_memory: false,
            ..default()
        });
        $passes_map.insert(pass, (layout, id));
    }};
}

/// Holds all compute pipelines and their shared bind group layouts.
#[derive(Resource)]
pub struct WaterPipelines {
    pub passes: HashMap<ComputePasses, (BindGroupLayoutDescriptor, CachedComputePipelineId)>,
}

impl FromWorld for WaterPipelines {
    fn from_world(world: &mut World) -> Self {
        let mut passes = HashMap::new();

        let asset_server  = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        compute_pipeline!(passes, ComputePasses::WaterInject, asset_server, pipeline_cache, (
            texture_storage_2d_array(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
            uniform_buffer::<WaterSettings>(false),
            uniform_buffer::<WaterDebugSettings>(false),
        ));

        compute_pipeline!(passes, ComputePasses::WaterAdvection, asset_server, pipeline_cache, (
            texture_2d_array(bevy::render::render_resource::TextureSampleType::Float { filterable: true }),
            sampler(bevy::render::render_resource::SamplerBindingType::Filtering),
            texture_storage_2d_array(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
            uniform_buffer::<WaterSettings>(false),
            texture_3d(bevy::render::render_resource::TextureSampleType::Uint),
            uniform_buffer::<VoxelSettings>(false),
            uniform_buffer::<WaterWaveGroupSpeeds>(false),
            texture_3d(bevy::render::render_resource::TextureSampleType::Float { filterable: false }),
        ));

        compute_pipeline!(passes, ComputePasses::WaterDiffusion, asset_server, pipeline_cache, (
            texture_2d_array(bevy::render::render_resource::TextureSampleType::Float { filterable: true }),
            sampler(bevy::render::render_resource::SamplerBindingType::Filtering),
            texture_storage_2d_array(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
            uniform_buffer::<WaterSettings>(false),
            uniform_buffer::<WaterWaveGroupSpeeds>(false),
        ));

        compute_pipeline!(passes, ComputePasses::WaterProfile, asset_server, pipeline_cache, (
            texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            uniform_buffer::<WaterSettings>(false),
        ));

        Self { passes }
    }
}

// ─── Bind Groups (rebuilt each frame because ping-pong flips) ─────────────────

#[derive(Resource)]
pub struct WaterBindGroups {
    pub bind_groups: HashMap<ComputePasses, BindGroup>,
    pub buffers: HashMap<Buffers, Buffer>,
}

pub fn queue_bind_groups(
    mut commands:  Commands,
    device:        Res<RenderDevice>,
    pipelines:     Res<WaterPipelines>,
    gpu:           Option<ResMut<WaterGpuResources>>,
    settings:      Res<WaterSettings>,
    debug_settings: Res<WaterDebugSettings>,
    voxel_settings: Res<VoxelSettings>,
    water_wave_group_speeds:  Res<WaterWaveGroupSpeeds>,
    existing_bgs:  Option<ResMut<WaterBindGroups>>,
    textures:      Res<GpuTextures>,
    ping_pong:     Res<WaterPingPong>,
    render_images: Res<RenderAssets<GpuImage>>,
    queue:         Res<bevy::render::renderer::RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(gpu) = gpu else { return };

    let lookup = |key: GpuTexture| -> Option<&GpuImage> {
        textures.get(key).and_then(|h| render_images.get(h))
    };

    let Some(amp_a) = lookup(GpuTexture::AmplitudeA) else { return };
    let Some(amp_b) = lookup(GpuTexture::AmplitudeB) else { return };
    let Some(profile) = lookup(GpuTexture::Profile) else { return };
    let Some(voxel_gpu_image) = lookup(GpuTexture::Voxel) else { return };
    let Some(sdf_gpu_image) = lookup(GpuTexture::Sdf) else { return };

    let voxel_texture_view = &voxel_gpu_image.texture_view;
    let sdf_texture_view = &sdf_gpu_image.texture_view;

    // ── Create or update the uniform buffers ────────────────────────────────
    let buffers: HashMap<Buffers, Buffer> = if let Some(bgs) = existing_bgs {
        // Update existing buffers in-place.
        queue.write_buffer(&bgs.buffers[&Buffers::Settings], 0, &encode_uniform(&*settings));
        queue.write_buffer(&bgs.buffers[&Buffers::DebugSettings], 0, &encode_uniform(&*debug_settings));
        queue.write_buffer(&bgs.buffers[&Buffers::VoxelSettings], 0, &encode_uniform(&*voxel_settings));
        // WaterWaveGroupSpeeds doesn't change at runtime.

        bgs.buffers.clone()
    } else {
        // First frame – create buffers from scratch.
        let mut bufs = HashMap::new();

        let mk = |label, data: Vec<u8>| device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(label),
            contents: &data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        bufs.insert(Buffers::Settings, mk("water_settings_buf", encode_uniform(&*settings)));
        bufs.insert(Buffers::DebugSettings, mk("water_debug_settings_buf", encode_uniform(&*debug_settings)));
        bufs.insert(Buffers::VoxelSettings, mk("water_voxel_settings_buf", encode_uniform(&*voxel_settings)));

        // WaterWaveGroupSpeeds uses dynamic alignment
        let mut dyn_buf = encase::DynamicUniformBuffer::new_with_alignment(Vec::<u8>::new(), 32);
        dyn_buf.write(&*water_wave_group_speeds).expect("ShaderType write failed");
        bufs.insert(Buffers::WaterWaveGroupSpeeds, mk("water_wave_group_speeds_buf", dyn_buf.into_inner()));

        bufs
    };

    let (amp_read, amp_write) = if ping_pong.flipped {
        (&amp_b.texture_view, &amp_a.texture_view)
    } else {
        (&amp_a.texture_view, &amp_b.texture_view)
    };

    // ── Helper: look up layout for a pass ────────────────────────────────────
    let layout_for = |pass: ComputePasses| -> &BindGroupLayoutDescriptor {
        &pipelines.passes[&pass].0
    };

    let settings_buf        = &buffers[&Buffers::Settings];
    let debug_settings_buf  = &buffers[&Buffers::DebugSettings];
    let voxel_settings_buf  = &buffers[&Buffers::VoxelSettings];
    let wave_speeds_buf     = &buffers[&Buffers::WaterWaveGroupSpeeds];

    let mut bind_groups = HashMap::new();

    bind_groups.insert(ComputePasses::WaterInject, device.create_bind_group(
        "inject_bg",
        &pipeline_cache.get_bind_group_layout(layout_for(ComputePasses::WaterInject)),
        &BindGroupEntries::sequential((
            amp_read,
            settings_buf.as_entire_binding(),
            debug_settings_buf.as_entire_binding(),
        )),
    ));

    bind_groups.insert(ComputePasses::WaterAdvection, device.create_bind_group(
        "advection_bg",
        &pipeline_cache.get_bind_group_layout(layout_for(ComputePasses::WaterAdvection)),
        &BindGroupEntries::sequential((
            amp_read,
            &gpu.clamp_linear_sampler,
            amp_write,
            settings_buf.as_entire_binding(),
            voxel_texture_view,
            voxel_settings_buf.as_entire_binding(),
            wave_speeds_buf.as_entire_binding(),
            sdf_texture_view,
        )),
    ));

    bind_groups.insert(ComputePasses::WaterDiffusion, device.create_bind_group(
        "diffusion_bg",
        &pipeline_cache.get_bind_group_layout(layout_for(ComputePasses::WaterDiffusion)),
        &BindGroupEntries::sequential((
            amp_write,
            &gpu.clamp_linear_sampler,
            amp_read,
            settings_buf.as_entire_binding(),
            wave_speeds_buf.as_entire_binding(),
        )),
    ));

    bind_groups.insert(ComputePasses::WaterProfile, device.create_bind_group(
        "profile_bg",
        &pipeline_cache.get_bind_group_layout(layout_for(ComputePasses::WaterProfile)),
        &BindGroupEntries::sequential((
            &profile.texture_view,
            settings_buf.as_entire_binding(),
        )),
    ));

    commands.insert_resource(WaterBindGroups { bind_groups, buffers });
}