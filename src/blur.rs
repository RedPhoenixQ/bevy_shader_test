use std::borrow::Cow;

use bevy::{
    prelude::*,
    render::{
        render_graph,
        render_resource::{
            CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache,
        },
        renderer::RenderContext,
    },
};

use crate::{GameOfLifeImageBindGroup, GameOfLifePipeline};
// #[derive(Resource)]
pub struct BlurPipeline {
    run_pipeline: CachedComputePipelineId,
}

impl FromWorld for BlurPipeline {
    fn from_world(world: &mut World) -> Self {
        let texture_bind_group_layout = &world
            .resource::<GameOfLifePipeline>()
            .texture_bind_group_layout
            .clone();

        let shader = world.resource::<AssetServer>().load("shaders/utils.wgsl");
        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let run_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![texture_bind_group_layout.clone()]),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("blur"),
        });

        BlurPipeline { run_pipeline }
    }
}

enum BlurState {
    Loading,
    Init,
    Update,
}

pub struct BlurNode {
    state: BlurState,
}

impl Default for BlurNode {
    fn default() -> Self {
        Self {
            state: BlurState::Loading,
        }
    }
}

impl render_graph::Node for BlurNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<BlurPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            BlurState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.run_pipeline)
                {
                    self.state = BlurState::Init;
                }
            }
            BlurState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.run_pipeline)
                {
                    self.state = BlurState::Update;
                }
            }
            BlurState::Update => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let texture_bind_group = &world.resource::<GameOfLifeImageBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<BlurPipeline>();
        let settings = world.resource::<crate::SimSettings>();

        // select the pipeline based on the current state
        match settings.state {
            crate::SimState::Playing => {
                let mut pass = render_context
                    .command_encoder
                    .begin_compute_pass(&ComputePassDescriptor::default());

                pass.set_bind_group(0, texture_bind_group, &[]);

                let run_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.run_pipeline)
                    .unwrap();
                pass.set_pipeline(run_pipeline);
                pass.dispatch_workgroups(
                    crate::SIZE.0 / crate::WORKGROUP_SIZE,
                    crate::SIZE.1 / crate::WORKGROUP_SIZE,
                    1,
                );
            }
            _ => {}
        }

        Ok(())
    }
}
