use bevy::{
    render::{
        render_graph::{self, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::{PipelineCache, ComputePassDescriptor},
        renderer::RenderContext,
    },
    prelude::*,
};

use super::{GRID_SIZE, NUM_DIRECTIONS, PROFILE_SIZE, pipeline::{WaterPipelines, WaterBindGroups}};

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
        let (Some(adv_pl), Some(prof_pl), Some(hgt_pl)) = (
            pipeline_cache.get_compute_pipeline(pipelines.advection_id),
            pipeline_cache.get_compute_pipeline(pipelines.profile_id),
            pipeline_cache.get_compute_pipeline(pipelines.height_id),
        ) else {
            return Ok(());
        };

        // ── Workgroup counts ───────────────────────────────────────────────
        // Each shader uses @workgroup_size(8, 8, 1) over the grid / profile.
        let wg_grid    = (GRID_SIZE.div_ceil(8), GRID_SIZE.div_ceil(8), NUM_DIRECTIONS);
        let wg_profile = (PROFILE_SIZE.div_ceil(64), 1, 1);
        let wg_height  = (GRID_SIZE.div_ceil(8), GRID_SIZE.div_ceil(8), 1);

        let encoder = ctx.command_encoder();

        // ── Pass 1 : Profile buffer ────────────────────────────────────────
        // Build Ψ(p) from the Phillips spectrum.
        // Done before advection so height reconstruction uses an up-to-date profile.
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label:            Some("water_profile"),
                timestamp_writes: None,
            });
            pass.set_pipeline(prof_pl);
            pass.set_bind_group(0, &bgs.profile, &[]);
            pass.dispatch_workgroups(wg_profile.0, wg_profile.1, wg_profile.2);
        }

        // ── Pass 2 : Advection ─────────────────────────────────────────────
        // Semi-Lagrangian advection of A(x, y, θ_d).
        // Dispatched once per direction in the Z dimension of the workgroup.
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label:            Some("water_advection"),
                timestamp_writes: None,
            });
            pass.set_pipeline(adv_pl);
            pass.set_bind_group(0, &bgs.advection, &[]);
            pass.dispatch_workgroups(wg_grid.0, wg_grid.1, wg_grid.2);
        }

        // ── Pass 3 : Height reconstruction ─────────────────────────────────
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
