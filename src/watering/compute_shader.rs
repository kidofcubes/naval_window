use std::collections::HashMap;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel};
use bevy::render::render_resource::{encase, AddressMode, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer, BufferInitDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, FilterMode, PipelineCache, Sampler, SamplerDescriptor, ShaderStages, StorageTextureAccess, TextureFormat};
use bevy::render::render_resource::binding_types::{sampler, texture_2d_array, texture_3d, texture_storage_2d, texture_storage_2d_array, uniform_buffer};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use strum::AsRefStr;
use bevy::render::render_resource::binding_types::texture_2d;
use crate::watering::{FftCascadeHandles, FftOceanHandles, FftOceanSettings, FftOceanSettingsArray, VoxelGridHandle, VoxelSettings, WaterDebugSettings, WaterSettings, WaterWaveGroupSpeeds, DIRECTIONS, FFT_CASCADE_COUNT, GRID_X_SIZE, GRID_Z_SIZE, PROFILE_BUFFER_LENGTH, WaterSimulationHandles};

// ─── Pass / Buffer Enums ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, AsRefStr)]
#[strum(serialize_all = "snake_case")]
pub enum ComputePasses {
    WaterInject,
    WaterAdvection,
    WaterDiffusion,
    WaterProfile,
    FftTimeEvolve,
    FftIfftRows,
    FftIfftCols,
    FftIfftRowsSlope,
    FftIfftColsSlope,
}

impl ComputePasses {
    pub fn wgsl_path(&self) -> String {
        match self {
            ComputePasses::FftIfftRowsSlope => "fft_ifft_rows.wgsl".to_string(),
            other => format!("{}.wgsl", other.as_ref()),
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
    FftOceanSettings0,
    FftOceanSettings1,
    FftOceanSettings2,
}

const FFT_SETTINGS_BUFFERS: [Buffers; FFT_CASCADE_COUNT] = [
    Buffers::FftOceanSettings0,
    Buffers::FftOceanSettings1,
    Buffers::FftOceanSettings2,
];

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
        let wg_grid    = (GRID_X_SIZE.div_ceil(32) as u32, GRID_Z_SIZE.div_ceil(32) as u32, DIRECTIONS as u32);
        let wg_profile = (PROFILE_BUFFER_LENGTH.div_ceil(128) as u32, 1, 1);

        let passes_to_run = [
            (ComputePasses::WaterInject, wg_grid),
            (ComputePasses::WaterAdvection, wg_grid),
            (ComputePasses::WaterDiffusion, wg_grid),
            (ComputePasses::WaterProfile, wg_profile),
        ];

        // ── FFT Ocean workgroup counts ───────────────────────────────────────
        let fft_size = GRID_X_SIZE as u32; // 256
        let wg_fft_grid = (fft_size.div_ceil(16), fft_size.div_ceil(16), 1);
        let wg_fft_rows = (1, fft_size, 1); // one workgroup per row
        let wg_fft_cols = (1, fft_size, 1); // one workgroup per column

        // Resolve all pipelines + bind groups; bail if any aren't ready.
        let mut resolved = Vec::with_capacity(passes_to_run.len() + FFT_CASCADE_COUNT * 5);
        for (pass, wg) in &passes_to_run {
            let Some((_, pipeline_id)) = pipelines.passes.get(pass) else { return Ok(()) };
            let Some(pipeline) = pipeline_cache.get_compute_pipeline(*pipeline_id) else { return Ok(()) };
            let Some(bg) = bgs.bind_groups.get(pass) else { return Ok(()) };
            resolved.push((pass.as_ref(), pipeline, bg, *wg));
        }

        // FFT passes run once per cascade (bind groups are per-cascade)
        let fft_pass_order = [
            (ComputePasses::FftTimeEvolve,    wg_fft_grid),
            (ComputePasses::FftIfftRows,      wg_fft_rows),
            (ComputePasses::FftIfftCols,      wg_fft_cols),
            (ComputePasses::FftIfftRowsSlope, wg_fft_rows),
            (ComputePasses::FftIfftColsSlope, wg_fft_cols),
        ];
        // Resolve pipelines once (shared across all cascades)
        let fft_pipelines: Vec<Option<&_>> = fft_pass_order.iter().map(|(pass, _)| {
            let (_, pipeline_id) = pipelines.passes.get(pass)?;
            pipeline_cache.get_compute_pipeline(*pipeline_id)
        }).collect();

        for cascade_bgs in &bgs.fft_cascade_bgs {
            let cascade_bind_groups = [
                &cascade_bgs.time_evolve,
                &cascade_bgs.ifft_rows,
                &cascade_bgs.ifft_cols,
                &cascade_bgs.ifft_rows_slope,
                &cascade_bgs.ifft_cols_slope,
            ];
            for (idx, (_, wg)) in fft_pass_order.iter().enumerate() {
                let Some(pipeline) = fft_pipelines[idx] else { continue };
                resolved.push(("fft", pipeline, cascade_bind_groups[idx], *wg));
            }
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
            push_constant_ranges:            vec![],
            shader:                          $asset_server.load(pass.wgsl_path()),
            shader_defs:                     vec![],
            entry_point:                     Some("main".into()),
            zero_initialize_workgroup_memory: false,
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

        // ── FFT Ocean Passes ─────────────────────────────────────────────────
        compute_pipeline!(passes, ComputePasses::FftTimeEvolve, asset_server, pipeline_cache, (
            texture_2d(bevy::render::render_resource::TextureSampleType::Float { filterable: false }),  // initial_spectrum (Rg32Float)
            texture_2d(bevy::render::render_resource::TextureSampleType::Float { filterable: false }),  // initial_spectrum_conj
            texture_2d(bevy::render::render_resource::TextureSampleType::Float { filterable: false }),  // angular_frequency
            texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),              // evolved_out
            uniform_buffer::<FftOceanSettings>(false),
            texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),              // slope_out
        ));

        compute_pipeline!(passes, ComputePasses::FftIfftRows, asset_server, pipeline_cache, (
            texture_2d(bevy::render::render_resource::TextureSampleType::Float { filterable: false }),  // spectrum_in
            texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),              // spectrum_out
        ));

        compute_pipeline!(passes, ComputePasses::FftIfftCols, asset_server, pipeline_cache, (
            texture_2d(bevy::render::render_resource::TextureSampleType::Float { filterable: false }),  // spectrum_in (after row IFFT)
            texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),               // height_out
        ));

        compute_pipeline!(passes, ComputePasses::FftIfftRowsSlope, asset_server, pipeline_cache, (
            texture_2d(bevy::render::render_resource::TextureSampleType::Float { filterable: false }),  // slope_spectrum
            texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),              // evolved_a (reused)
        ));

        compute_pipeline!(passes, ComputePasses::FftIfftColsSlope, asset_server, pipeline_cache, (
            texture_2d(bevy::render::render_resource::TextureSampleType::Float { filterable: false }),  // evolved_a (row-IFFT'd slope)
            texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),              // slope_field
        ));

        Self { passes }
    }
}

// ─── Bind Groups (rebuilt each frame because ping-pong flips) ─────────────────

pub struct FftCascadeBindGroups {
    pub time_evolve: BindGroup,
    pub ifft_rows: BindGroup,
    pub ifft_cols: BindGroup,
    pub ifft_rows_slope: BindGroup,
    pub ifft_cols_slope: BindGroup,
}

#[derive(Resource)]
pub struct WaterBindGroups {
    pub bind_groups: HashMap<ComputePasses, BindGroup>,
    pub buffers: HashMap<Buffers, Buffer>,
    pub fft_cascade_bgs: Vec<FftCascadeBindGroups>,
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
    sim_handle:     Option<ResMut<WaterSimulationHandles>>,
    fft_handles:   Option<Res<FftOceanHandles>>,
    fft_settings_array: Option<Res<FftOceanSettingsArray>>,
    render_images: Res<RenderAssets<GpuImage>>,
    queue:         Res<bevy::render::renderer::RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(gpu) = gpu else { return };
    let Some(sim_handle) = sim_handle else { return };

    let Some(amp_a) = render_images.get(&sim_handle.amplitude_texture_a) else { return };
    let Some(amp_b) = render_images.get(&sim_handle.amplitude_texture_b) else { return };
    let Some(profile_buffer) = render_images.get(&sim_handle.profile_buffer) else { return };

    let Some(voxel_handle_inner) = &voxel_handle.voxel else { return };
    let Some(voxel_gpu_image) = render_images.get(voxel_handle_inner) else { return };
    let voxel_texture_view = &voxel_gpu_image.texture_view;

    let Some(sdf_handle) = &voxel_handle.sdf else { return };
    let Some(sdf_gpu_image) = render_images.get(sdf_handle) else { return };
    let sdf_texture_view = &sdf_gpu_image.texture_view;

    // ── Helper: encode a ShaderType into a uniform buffer ────────────────────
    macro_rules! encode_uniform {
        ($data:expr) => {{
            let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
            buf.write($data).expect("ShaderType write failed");
            buf.into_inner()
        }};
    }

    // ── Create or update the uniform buffers ────────────────────────────────
    let buffers: HashMap<Buffers, Buffer> = if let Some(bgs) = existing_bgs {
        // Update existing buffers in-place.
        queue.write_buffer(&bgs.buffers[&Buffers::Settings], 0, &encode_uniform!(&*settings));
        queue.write_buffer(&bgs.buffers[&Buffers::DebugSettings], 0, &encode_uniform!(&*debug_settings));
        queue.write_buffer(&bgs.buffers[&Buffers::VoxelSettings], 0, &encode_uniform!(&*voxel_settings));
        if let Some(ref fft_sa) = fft_settings_array {
            for i in 0..FFT_CASCADE_COUNT {
                queue.write_buffer(&bgs.buffers[&FFT_SETTINGS_BUFFERS[i]], 0, &encode_uniform!(&fft_sa.cascades[i]));
            }
        }

        bgs.buffers.clone()
    } else {
        // First frame – create buffers from scratch.
        let mut bufs = HashMap::new();

        let mk = |label, data: Vec<u8>| device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(label),
            contents: &data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        bufs.insert(Buffers::Settings, mk("water_settings_buf", encode_uniform!(&*settings)));
        bufs.insert(Buffers::DebugSettings, mk("water_debug_settings_buf", encode_uniform!(&*debug_settings)));
        bufs.insert(Buffers::VoxelSettings, mk("water_voxel_settings_buf", encode_uniform!(&*voxel_settings)));

        // WaterWaveGroupSpeeds uses dynamic alignment
        let mut dyn_buf = encase::DynamicUniformBuffer::new_with_alignment(Vec::<u8>::new(), 32);
        dyn_buf.write(&*water_wave_group_speeds).expect("ShaderType write failed");
        bufs.insert(Buffers::WaterWaveGroupSpeeds, mk("water_wave_group_speeds_buf", dyn_buf.into_inner()));

        if let Some(ref fft_sa) = fft_settings_array {
            for i in 0..FFT_CASCADE_COUNT {
                bufs.insert(FFT_SETTINGS_BUFFERS[i], mk(
                    Box::leak(format!("fft_ocean_settings_buf_{i}").into_boxed_str()),
                    encode_uniform!(&fft_sa.cascades[i])
                ));
            }
        }

        bufs
    };

    let (amp_read, amp_write) = if sim_handle.flipped {
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
            &profile_buffer.texture_view,
            settings_buf.as_entire_binding(),
        )),
    ));

    // ── FFT Ocean Bind Groups (per cascade) ─────────────────────────────────
    let mut fft_cascade_bgs: Vec<FftCascadeBindGroups> = Vec::new();

    if let (Some(fft_h), Some(_fft_sa)) = (&fft_handles, &fft_settings_array) {
        for i in 0..FFT_CASCADE_COUNT {
            let c = &fft_h.cascades[i];
            let cascade_ready =
                render_images.get(&c.initial_spectrum).is_some()
                && render_images.get(&c.initial_spectrum_conj).is_some()
                && render_images.get(&c.angular_frequency).is_some()
                && render_images.get(&c.evolved_spectrum_a).is_some()
                && render_images.get(&c.evolved_spectrum_b).is_some()
                && render_images.get(&c.height_field).is_some()
                && render_images.get(&c.slope_spectrum).is_some()
                && render_images.get(&c.slope_field).is_some()
                && buffers.contains_key(&FFT_SETTINGS_BUFFERS[i]);

            if !cascade_ready { break; }

            let init_spec  = &render_images.get(&c.initial_spectrum).unwrap().texture_view;
            let init_conj  = &render_images.get(&c.initial_spectrum_conj).unwrap().texture_view;
            let ang_freq   = &render_images.get(&c.angular_frequency).unwrap().texture_view;
            let evolved_a  = &render_images.get(&c.evolved_spectrum_a).unwrap().texture_view;
            let evolved_b  = &render_images.get(&c.evolved_spectrum_b).unwrap().texture_view;
            let height_fld = &render_images.get(&c.height_field).unwrap().texture_view;
            let slope_spec = &render_images.get(&c.slope_spectrum).unwrap().texture_view;
            let slope_fld  = &render_images.get(&c.slope_field).unwrap().texture_view;
            let fft_settings_buf = &buffers[&FFT_SETTINGS_BUFFERS[i]];

            let time_evolve = device.create_bind_group(
                "fft_time_evolve_bg",
                &pipeline_cache.get_bind_group_layout(layout_for(ComputePasses::FftTimeEvolve)),
                &BindGroupEntries::sequential((
                    init_spec,
                    init_conj,
                    ang_freq,
                    evolved_a,
                    fft_settings_buf.as_entire_binding(),
                    slope_spec,
                )),
            );

            let ifft_rows = device.create_bind_group(
                "fft_ifft_rows_bg",
                &pipeline_cache.get_bind_group_layout(layout_for(ComputePasses::FftIfftRows)),
                &BindGroupEntries::sequential((evolved_a, evolved_b)),
            );

            let ifft_cols = device.create_bind_group(
                "fft_ifft_cols_bg",
                &pipeline_cache.get_bind_group_layout(layout_for(ComputePasses::FftIfftCols)),
                &BindGroupEntries::sequential((evolved_b, height_fld)),
            );

            let ifft_rows_slope = device.create_bind_group(
                "fft_ifft_rows_slope_bg",
                &pipeline_cache.get_bind_group_layout(layout_for(ComputePasses::FftIfftRowsSlope)),
                &BindGroupEntries::sequential((slope_spec, evolved_a)),
            );

            let ifft_cols_slope = device.create_bind_group(
                "fft_ifft_cols_slope_bg",
                &pipeline_cache.get_bind_group_layout(layout_for(ComputePasses::FftIfftColsSlope)),
                &BindGroupEntries::sequential((evolved_a, slope_fld)),
            );

            fft_cascade_bgs.push(FftCascadeBindGroups {
                time_evolve,
                ifft_rows,
                ifft_cols,
                ifft_rows_slope,
                ifft_cols_slope,
            });
        }
    }

    commands.insert_resource(WaterBindGroups { bind_groups, buffers, fft_cascade_bgs });
}