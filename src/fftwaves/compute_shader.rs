use std::collections::HashMap;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    encase, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    Buffer, BufferInitDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
    ComputePipelineDescriptor, PipelineCache, ShaderStages, StorageTextureAccess, TextureFormat,
};
use bevy::render::render_resource::binding_types::{texture_storage_2d_array, uniform_buffer};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use strum::AsRefStr;

use crate::fftwaves::{
    FftCascadesUniform, FftSettings, FftSpectrumParams,
    FftTexture, FftTextures,
    FFT_CASCADE_COUNT, FFT_GRID_SIZE, FFT_SPECTRUM_COUNT,
};
use crate::watering::shader_utils::encode_uniform;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, AsRefStr)]
#[strum(serialize_all = "snake_case")]
pub enum FftPasses {
    FftInitSpectrum,
    FftEvolveSpectrum,
    FftButterflyRow,
    FftButterflyCol,
    FftAssemble,
}

impl FftPasses {
    fn wgsl_path(self) -> String {
        format!("fftwaves/{}.wgsl", self.as_ref())
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, AsRefStr)]
pub enum FftBuffers {
    Settings,
    Cascades,
    Spectrum,
}

#[derive(Resource)]
pub struct FftPipelines {
    pub passes: HashMap<FftPasses, (BindGroupLayoutDescriptor, CachedComputePipelineId)>,
}

macro_rules! fft_pipeline {
    ($passes_map:expr, $pass:expr, $asset_server:expr, $pipeline_cache:expr, $entries:expr) => {{
        let pass: FftPasses = $pass;
        let label: &'static str = Box::leak(pass.as_ref().to_owned().into_boxed_str());
        let layout_label: &'static str = Box::leak(format!("{}_layout", label).into_boxed_str());
        let layout = BindGroupLayoutDescriptor::new(
            layout_label,
            &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, $entries),
        );
        let id = $pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                            Some(label.into()),
            layout:                           vec![layout.clone()],
            shader:                           $asset_server.load(pass.wgsl_path()),
            shader_defs:                      vec![],
            entry_point:                      Some("main".into()),
            zero_initialize_workgroup_memory: false,
            ..default()
        });
        $passes_map.insert(pass, (layout, id));
    }};
}

impl FromWorld for FftPipelines {
    fn from_world(world: &mut World) -> Self {
        let mut passes = HashMap::new();
        let asset_server   = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        fft_pipeline!(passes, FftPasses::FftInitSpectrum, asset_server, pipeline_cache, (
            texture_storage_2d_array(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
            texture_storage_2d_array(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
            texture_storage_2d_array(TextureFormat::R32Float,  StorageTextureAccess::WriteOnly),
            uniform_buffer::<FftCascadesUniform>(false),
            uniform_buffer::<FftSpectrumParams>(false),
        ));

        fft_pipeline!(passes, FftPasses::FftEvolveSpectrum, asset_server, pipeline_cache, (
            texture_storage_2d_array(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
            texture_storage_2d_array(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
            texture_storage_2d_array(TextureFormat::R32Float,  StorageTextureAccess::ReadOnly),
            texture_storage_2d_array(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
            uniform_buffer::<FftSettings>(false),
            uniform_buffer::<FftCascadesUniform>(false),
        ));

        fft_pipeline!(passes, FftPasses::FftButterflyRow, asset_server, pipeline_cache, (
            texture_storage_2d_array(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
            texture_storage_2d_array(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
        ));

        fft_pipeline!(passes, FftPasses::FftButterflyCol, asset_server, pipeline_cache, (
            texture_storage_2d_array(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
            texture_storage_2d_array(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
        ));

        fft_pipeline!(passes, FftPasses::FftAssemble, asset_server, pipeline_cache, (
            texture_storage_2d_array(TextureFormat::Rg32Float,   StorageTextureAccess::ReadOnly),
            texture_storage_2d_array(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            texture_storage_2d_array(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
        ));

        Self { passes }
    }
}

#[derive(Resource, Default)]
pub struct FftGpuResources;

pub fn prepare_gpu_resources(
    mut commands: Commands,
    existing: Option<Res<FftGpuResources>>,
) {
    if existing.is_some() {
        return;
    }
    commands.insert_resource(FftGpuResources);
}

#[derive(Resource)]
pub struct FftBindGroups {
    pub bind_groups: HashMap<FftPasses, BindGroup>,
    pub buffers:     HashMap<FftBuffers, Buffer>,
}

pub fn queue_bind_groups(
    mut commands:     Commands,
    device:           Res<RenderDevice>,
    queue:            Res<RenderQueue>,
    pipelines:        Res<FftPipelines>,
    settings:         Res<FftSettings>,
    cascades:         Res<FftCascadesUniform>,
    spectrum:         Res<FftSpectrumParams>,
    existing_bgs:     Option<ResMut<FftBindGroups>>,
    textures:         Res<FftTextures>,
    render_images:    Res<RenderAssets<GpuImage>>,
    pipeline_cache:   Res<PipelineCache>,
) {
    let lookup = |key: FftTexture| -> Option<&GpuImage> {
        textures.get(key).and_then(|h| render_images.get(h))
    };
    let Some(h0)      = lookup(FftTexture::H0)           else { return };
    let Some(h0c)     = lookup(FftTexture::H0Conj)       else { return };
    let Some(omega)   = lookup(FftTexture::Omega)        else { return };
    let Some(spec_a)  = lookup(FftTexture::SpectraA)     else { return };
    let Some(spec_b)  = lookup(FftTexture::SpectraB)     else { return };
    let Some(disp)    = lookup(FftTexture::Displacement) else { return };
    let Some(slope)   = lookup(FftTexture::Slope)        else { return };

    // ── Uniforms ──────────────────────────────────────────────────────────
    let buffers: HashMap<FftBuffers, Buffer> = if let Some(bgs) = existing_bgs {
        queue.write_buffer(&bgs.buffers[&FftBuffers::Settings], 0, &encode_uniform(&*settings));
        queue.write_buffer(&bgs.buffers[&FftBuffers::Cascades], 0, &encode_uniform(&*cascades));
        queue.write_buffer(&bgs.buffers[&FftBuffers::Spectrum], 0, &encode_uniform(&*spectrum));
        bgs.buffers.clone()
    } else {
        let mut bufs = HashMap::new();
        let mk = |label, data: Vec<u8>| device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(label),
            contents: &data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        bufs.insert(FftBuffers::Settings, mk("fft_settings", encode_uniform(&*settings)));
        bufs.insert(FftBuffers::Cascades, mk("fft_cascades", encode_uniform(&*cascades)));
        bufs.insert(FftBuffers::Spectrum, mk("fft_spectrum", encode_uniform(&*spectrum)));
        bufs
    };

    let layout_for = |pass: FftPasses| -> &BindGroupLayoutDescriptor {
        &pipelines.passes[&pass].0
    };

    let settings_buf = &buffers[&FftBuffers::Settings];
    let cascades_buf = &buffers[&FftBuffers::Cascades];
    let spectrum_buf = &buffers[&FftBuffers::Spectrum];

    let mut bind_groups = HashMap::new();

    bind_groups.insert(FftPasses::FftInitSpectrum, device.create_bind_group(
        "fft_init_bg",
        &pipeline_cache.get_bind_group_layout(layout_for(FftPasses::FftInitSpectrum)),
        &BindGroupEntries::sequential((
            &h0.texture_view,
            &h0c.texture_view,
            &omega.texture_view,
            cascades_buf.as_entire_binding(),
            spectrum_buf.as_entire_binding(),
        )),
    ));

    bind_groups.insert(FftPasses::FftEvolveSpectrum, device.create_bind_group(
        "fft_evolve_bg",
        &pipeline_cache.get_bind_group_layout(layout_for(FftPasses::FftEvolveSpectrum)),
        &BindGroupEntries::sequential((
            &h0.texture_view,
            &h0c.texture_view,
            &omega.texture_view,
            &spec_a.texture_view,
            settings_buf.as_entire_binding(),
            cascades_buf.as_entire_binding(),
        )),
    ));

    bind_groups.insert(FftPasses::FftButterflyRow, device.create_bind_group(
        "fft_butterfly_row_bg",
        &pipeline_cache.get_bind_group_layout(layout_for(FftPasses::FftButterflyRow)),
        &BindGroupEntries::sequential((
            &spec_a.texture_view, // in: result of evolve
            &spec_b.texture_view, // out: row-pass scratch
        )),
    ));

    bind_groups.insert(FftPasses::FftButterflyCol, device.create_bind_group(
        "fft_butterfly_col_bg",
        &pipeline_cache.get_bind_group_layout(layout_for(FftPasses::FftButterflyCol)),
        &BindGroupEntries::sequential((
            &spec_b.texture_view, // in: row-pass result
            &spec_a.texture_view, // out: final spatial-domain field
        )),
    ));

    bind_groups.insert(FftPasses::FftAssemble, device.create_bind_group(
        "fft_assemble_bg",
        &pipeline_cache.get_bind_group_layout(layout_for(FftPasses::FftAssemble)),
        &BindGroupEntries::sequential((
            &spec_a.texture_view,
            &disp.texture_view,
            &slope.texture_view,
        )),
    ));

    commands.insert_resource(FftBindGroups { bind_groups, buffers });
}

const WG_XY: u32 = 8;
fn cell_workgroups() -> (u32, u32, u32) {
    let n = FFT_GRID_SIZE as u32;
    (n.div_ceil(WG_XY), n.div_ceil(WG_XY), FFT_CASCADE_COUNT as u32)
}
fn butterfly_workgroups() -> (u32, u32, u32) {
    (1, FFT_GRID_SIZE as u32, (FFT_SPECTRUM_COUNT * FFT_CASCADE_COUNT) as u32)
}

pub fn render_pass(
    world: &World,
    mut ctx: RenderContext,
    mut last_init_signature: Local<Vec<u8>>,
) -> Result<()> {
    let pipeline_cache = world.resource::<PipelineCache>();
    let pipelines      = world.resource::<FftPipelines>();
    let Some(bgs)      = world.get_resource::<FftBindGroups>() else { return Ok(()) };

    // Re-run InitSpectrum whenever the JONSWAP params or cascade bands change.
    let spectrum = world.resource::<FftSpectrumParams>();
    let cascades = world.resource::<FftCascadesUniform>();
    let mut signature = encode_uniform(spectrum);
    signature.extend_from_slice(&encode_uniform(cascades));
    let needs_init = signature != *last_init_signature;

    let wg_cell = cell_workgroups();
    let wg_bfly = butterfly_workgroups();

    let mut passes_to_run: Vec<(FftPasses, (u32, u32, u32))> = Vec::with_capacity(5);
    if needs_init {
        passes_to_run.push((FftPasses::FftInitSpectrum, wg_cell));
    }
    passes_to_run.push((FftPasses::FftEvolveSpectrum, wg_cell));
    passes_to_run.push((FftPasses::FftButterflyRow,   wg_bfly));
    passes_to_run.push((FftPasses::FftButterflyCol,   wg_bfly));
    passes_to_run.push((FftPasses::FftAssemble,       wg_cell));

    // Resolve all pipelines/bind groups; bail if any aren't ready yet.
    let mut resolved = Vec::with_capacity(passes_to_run.len());
    for (pass, wg) in &passes_to_run {
        let Some((_, pid)) = pipelines.passes.get(pass) else { return Ok(()) };
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(*pid) else { return Ok(()) };
        let Some(bg)       = bgs.bind_groups.get(pass) else { return Ok(()) };
        resolved.push((pass.as_ref(), pipeline, bg, *wg));
    }

    let encoder = ctx.command_encoder();
    for (label, pipeline, bg, wg) in resolved {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(wg.0, wg.1, wg.2);
    }

    if needs_init {
        *last_init_signature = signature;
    }

    Ok(())
}
