//! A compute shader that simulates Conway's Game of Life.
//!
//! Compute shaders use the GPU for computing arbitrary information, that may be independent of what
//! is rendered to the screen.
mod blur;
mod color;
mod decay;

use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        RenderApp, RenderStage,
    },
    window::WindowDescriptor,
};
use bevy_egui::{
    egui::{color_picker::color_edit_button_srgba, Button, Checkbox, Color32, ComboBox, Slider},
    EguiContext, EguiPlugin,
};
// use bevy_midi::{Midi, MidiRawData, MidiSettings};
use std::{borrow::Cow, ops::RangeInclusive};

const ZOOM: f32 = 1.0;
// pub const SIZE: (u32, u32) = (3440, 1440);
pub const WORKGROUP_SIZE: u32 = 16;
pub const GAME_WORKGROUP_SIZE: u32 = 512;
pub const NUM_AGENTS: u32 = 250000;
fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(WindowDescriptor {
            // uncomment for unthrottled FPS
            // present_mode: bevy::window::PresentMode::Fifo,
            mode: bevy::window::WindowMode::BorderlessFullscreen,
            ..default()
        })
        .add_plugins(DefaultPlugins)
        // .add_plugin(Midi)
        .add_plugin(GameOfLifeComputePlugin)
        .add_startup_system(setup)
        .add_system(update_params)
        .add_plugin(EguiPlugin)
        .add_system(ui_params)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, windows: Res<Windows>) {
    let window = windows.primary();

    let width = window.width().ceil() as u32
        + (WORKGROUP_SIZE - (window.width().ceil() as u32 % WORKGROUP_SIZE));
    let height = window.height().ceil() as u32
        + (WORKGROUP_SIZE - (window.height().ceil() as u32 % WORKGROUP_SIZE));

    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image = images.add(image);

    let mut image_second = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8Unorm,
    );
    image_second.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image_second = images.add(image_second);

    commands.spawn_bundle(SpriteBundle {
        sprite: Sprite {
            custom_size: Some(Vec2::new(width as f32, height as f32)),
            ..default()
        },
        texture: image.clone(),
        ..default()
    });
    commands.spawn_bundle(Camera2dBundle {
        transform: Transform {
            scale: Vec3 {
                x: ZOOM,
                y: ZOOM,
                z: ZOOM,
            },
            ..default()
        },
        ..default()
    });

    commands.insert_resource(GameOfLifeImage(image));

    commands.insert_resource(GameOfLifeImageSecond(image_second));

    let randomizable_array = RandArray {
        array: vec![
            // RandInfo::new(
            //     RandomizableParams::DecayRate,
            //     RandParams {
            //         pow: 4.0,
            //         start: 0.75,
            //         end: 5.0,
            //     },
            // )
            RandInfo::new(
                RandomizableParams::MoveSpeed,
                RandParams {
                    modifier: Some(|r| r.powf(1.2)),
                    start: 50.0,
                    end: 300.0,
                },
            ),
            RandInfo::new(
                RandomizableParams::TurnSpeed,
                RandParams {
                    modifier: None,
                    start: 1.0,
                    end: 10.0,
                },
            ),
            RandInfo::new(
                RandomizableParams::SensorAngleSpacing,
                RandParams {
                    modifier: None,
                    start: 20.0,
                    end: 355.0,
                },
            ),
            RandInfo::new(
                RandomizableParams::SensorOffsetDistance,
                RandParams {
                    modifier: Some(|r| r.powf(2.0)),
                    start: 20.0,
                    end: 400.0,
                },
            ),
        ],
    };

    commands.insert_resource(randomizable_array);

    let sim_params = SimParams {
        width,
        height,
        mode: SimSpawnMode::CircleIn,
        trail_weight: 0.75,
        decay_rate: 0.3,
        time: 0.01,
        delta: 0.01,
        salt: rand::random::<u32>(),
        move_speed: 100.0,
        turn_speed: 20.0,
        sensor_angle_spacing: 30.0,
        sensor_offset_distance: 60.0,
        sensor_size: 1,
        blur_mask: Color32::from_rgb(255, 240, 0),
        color: Color32::from_rgb(255, 255, 255),
    };

    commands.insert_resource(sim_params);

    let sim_settings = SimSettings {
        width,
        height,
        randomize: false,
        state: SimState::Initialize,
        params_change_per_frame: 0.01,
    };

    commands.insert_resource(sim_settings);

    commands.insert_resource(EguiState { all_visible: true })
}

#[derive(Debug, Clone, Copy)]
struct SimSettings {
    width: u32,
    height: u32,
    randomize: bool,
    state: SimState,
    params_change_per_frame: f32,
}

#[derive(Debug, Clone, Copy)]
enum SimState {
    Initialize,
    Playing,
    Paused,
}

impl ExtractResource for SimSettings {
    type Source = SimSettings;

    fn extract_resource(settings: &Self::Source) -> Self {
        *settings
    }
}

struct RandParams {
    modifier: Option<fn(f32) -> f32>,
    start: f32,
    end: f32,
}

impl RandParams {
    fn random(&self) -> f32 {
        let mut random = rand::random::<f32>();

        match self.modifier {
            Some(m) => random = m(random),
            None => {}
        }

        random * self.end % (self.end - self.start) + self.start
    }
}

#[allow(dead_code)]
enum RandomizableParams {
    DecayRate,
    MoveSpeed,
    TurnSpeed,
    SensorAngleSpacing,
    SensorOffsetDistance,
}

struct RandInfo {
    index: RandomizableParams,
    params: RandParams,
    value: f32,
    step: f32,
    changing: bool,
    target: bool,
}

impl RandInfo {
    fn new(index: RandomizableParams, params: RandParams) -> Self {
        Self {
            index,
            params,
            changing: false,
            step: 0.0,
            target: false,
            value: 0.0,
        }
    }
}

struct RandArray {
    array: Vec<RandInfo>,
}

struct EguiState {
    all_visible: bool,
}

fn update_params(
    mut sim_params: ResMut<SimParams>,
    mut rand_array: ResMut<RandArray>,
    time: Res<Time>,
    settings: Res<SimSettings>,
) {
    if !settings.randomize || settings.params_change_per_frame == 0.0 {
        return;
    }
    let change = time.delta_seconds() * settings.params_change_per_frame;

    for mut param in rand_array.array.iter_mut() {
        if !param.changing {
            let new_value = param.params.random();
            param.step = new_value - param.value;
            param.target = new_value < param.value;
            param.value = new_value;
            param.changing = true;
        }

        let mut current = 0.0;

        match param.index {
            RandomizableParams::DecayRate => {
                sim_params.decay_rate += param.step * change;
                current = sim_params.decay_rate;
            }
            RandomizableParams::MoveSpeed => {
                sim_params.move_speed += param.step * change;
                current = sim_params.move_speed;
            }
            RandomizableParams::TurnSpeed => {
                sim_params.turn_speed += param.step * change;
                current = sim_params.turn_speed;
            }
            RandomizableParams::SensorAngleSpacing => {
                sim_params.sensor_angle_spacing += param.step * change;
                current = sim_params.sensor_angle_spacing;
            }
            RandomizableParams::SensorOffsetDistance => {
                sim_params.sensor_offset_distance += param.step * change;
                current = sim_params.sensor_offset_distance;
            }
        }

        if (current < param.value) == param.target
        // || (current < param.params.start && param.step.is_sign_negative())
        // || (current > param.params.end && param.step.is_sign_positive())
        {
            param.changing = false
        }
    }
}

fn ui_params(
    mut egui_context: ResMut<EguiContext>,
    mut sim_params: ResMut<SimParams>,
    mut sim_settings: ResMut<SimSettings>,
    mut egui_state: ResMut<EguiState>,
    keys: Res<Input<KeyCode>>,
    rand_array: Res<RandArray>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        egui_state.all_visible = !egui_state.all_visible
    }

    if !egui_state.all_visible {
        return;
    }

    use bevy_egui::*;

    egui::Window::new("Settings").show(egui_context.ctx_mut(), |ui| {
        if ui.add(Button::new("Close")).clicked() {
            panic!("Application closed via menu")
        }
        if ui.add(Button::new("Play/Pause")).clicked() {
            match sim_settings.state {
                SimState::Playing => sim_settings.state = SimState::Paused,
                _ => sim_settings.state = SimState::Playing,
            }
        }
        ComboBox::from_label("Spawn Mode")
            .selected_text(format!("{:?}", sim_params.mode))
            .show_ui(ui, |ui| {
                if ui
                    .selectable_value(&mut sim_params.mode, SimSpawnMode::CenterOut, "Center Out")
                    .clicked()
                {
                    sim_settings.state = SimState::Initialize;
                };
                if ui
                    .selectable_value(&mut sim_params.mode, SimSpawnMode::CircleIn, "Circle In")
                    .clicked()
                {
                    sim_settings.state = SimState::Initialize;
                };
                if ui
                    .selectable_value(
                        &mut sim_params.mode,
                        SimSpawnMode::FullscreenRandom,
                        "Fullscreen Random",
                    )
                    .clicked()
                {
                    sim_settings.state = SimState::Initialize;
                };
            });
        ui.add(Checkbox::new(
            &mut sim_settings.randomize,
            "Randomize Params",
        ));
        if ui.add(Button::new("Randomize Params")).clicked() {
            for param in rand_array.array.iter() {
                match param.index {
                    RandomizableParams::DecayRate => sim_params.decay_rate = param.params.random(),
                    RandomizableParams::MoveSpeed => sim_params.move_speed = param.params.random(),
                    RandomizableParams::TurnSpeed => sim_params.turn_speed = param.params.random(),
                    RandomizableParams::SensorAngleSpacing => {
                        sim_params.sensor_angle_spacing = param.params.random()
                    }
                    RandomizableParams::SensorOffsetDistance => {
                        sim_params.sensor_offset_distance = param.params.random()
                    }
                }
            }
        }
        ui.add(
            Slider::new(
                &mut sim_settings.params_change_per_frame,
                RangeInclusive::<f32>::new(0.0, 10.0),
            )
            .text("params_change_per_frame")
            .step_by(0.001),
        );
    });

    egui::Window::new("Params").show(egui_context.ctx_mut(), |ui| {
        color_edit_button_srgba(ui, &mut sim_params.color, egui::color_picker::Alpha::Opaque);

        color_edit_button_srgba(
            ui,
            &mut sim_params.blur_mask,
            egui::color_picker::Alpha::Opaque,
        );

        ui.add(
            Slider::new(
                &mut sim_params.decay_rate,
                RangeInclusive::<f32>::new(0.01, 5.0),
            )
            .text("decay_rate"),
        );
        ui.add(
            Slider::new(
                &mut sim_params.move_speed,
                RangeInclusive::<f32>::new(10.0, 1000.0),
            )
            .text("move_speed"),
        );
        ui.add(
            Slider::new(
                &mut sim_params.turn_speed,
                RangeInclusive::<f32>::new(0.1, 100.0),
            )
            .text("turn_speed"),
        );
        ui.add(
            Slider::new(
                &mut sim_params.trail_weight,
                RangeInclusive::<f32>::new(0.1, 1.2),
            )
            .text("trail_weight"),
        );
        ui.add(
            Slider::new(
                &mut sim_params.sensor_size,
                RangeInclusive::<u32>::new(1, 10),
            )
            .text("sensor_size"),
        );
        ui.add(
            Slider::new(
                &mut sim_params.sensor_angle_spacing,
                RangeInclusive::<f32>::new(1.0, 360.0),
            )
            .text("sensor_angle_spacing"),
        );
        ui.add(
            Slider::new(
                &mut sim_params.sensor_offset_distance,
                RangeInclusive::<f32>::new(1.0, 1000.0),
            )
            .text("sensor_offset_distance"),
        );
    });
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Debug)]
struct Agent {
    position: [f32; 2],
    angle: f32,
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
struct SimParams {
    color: Color32,
    blur_mask: Color32,
    width: u32,
    height: u32,
    mode: SimSpawnMode,
    trail_weight: f32,
    decay_rate: f32,
    time: f32,
    delta: f32,
    salt: u32,
    move_speed: f32,
    turn_speed: f32,
    sensor_angle_spacing: f32,
    sensor_offset_distance: f32,
    sensor_size: u32,
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone, PartialEq)]
enum SimSpawnMode {
    CenterOut = 0,
    CircleIn = 1,
    FullscreenRandom = 2,
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParamsExport {
    color: [f32; 4],
    blur_mask: [f32; 4],
    width: u32,
    height: u32,
    mode: u32,
    trail_weight: f32,
    decay_rate: f32,
    time: f32,
    delta: f32,
    salt: u32,
    move_speed: f32,
    turn_speed: f32,
    sensor_angle_spacing: f32,
    sensor_offset_distance: f32,
    sensor_size: u32,
}

impl ExtractResource for SimParams {
    type Source = SimParams;

    fn extract_resource(params: &Self::Source) -> Self {
        *params
    }
}

pub struct GameOfLifeComputePlugin;

struct SimMeta {
    agents_buffer: Buffer,
    params_buffer: Buffer,
}

struct ExtractedTime {
    seconds_since_startup: f32,
    delta_time: f32,
}

impl ExtractResource for ExtractedTime {
    type Source = Time;

    fn extract_resource(time: &Self::Source) -> Self {
        Self {
            seconds_since_startup: time.seconds_since_startup() as f32,
            delta_time: time.delta_seconds(),
        }
    }
}

impl Plugin for GameOfLifeComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.

        let render_device = app.world.resource::<RenderDevice>();

        let agents_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("Agents Buffer"),
            size: std::mem::size_of::<[Agent; NUM_AGENTS as usize]>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let params_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("Params buffer"),
            size: (std::mem::size_of::<SimParamsExport>() + 16
                - (std::mem::size_of::<SimParamsExport>() % 16)) as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        app.add_plugin(ExtractResourcePlugin::<GameOfLifeImage>::default())
            .add_plugin(ExtractResourcePlugin::<GameOfLifeImageSecond>::default())
            .add_plugin(ExtractResourcePlugin::<ExtractedTime>::default())
            .add_plugin(ExtractResourcePlugin::<SimSettings>::default())
            .add_plugin(ExtractResourcePlugin::<SimParams>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<GameOfLifePipeline>()
            .add_system_to_stage(RenderStage::Queue, queue_bind_group)
            .init_resource::<blur::BlurPipeline>()
            .init_resource::<decay::DecayPipeline>()
            .init_resource::<color::ColorPipeline>()
            // .add_system_to_stage(RenderStage::Queue, queue_bind_group)
            .insert_resource(SimMeta {
                agents_buffer,
                params_buffer,
            })
            .add_system_to_stage(RenderStage::Prepare, prepare_params);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("game_of_life", GameOfLifeNode::default());
        render_graph
            .add_node_edge(
                "game_of_life",
                bevy::render::main_graph::node::CAMERA_DRIVER,
            )
            .unwrap();
        render_graph.add_node("blur", blur::BlurNode::default());
        render_graph.add_node_edge("blur", "game_of_life").unwrap();
        render_graph.add_node("decay", decay::DecayNode::default());
        render_graph.add_node_edge("decay", "blur").unwrap();
        // render_graph.add_node("color", color::ColorNode::default());
        // render_graph.add_node_edge("game_of_life", "color").unwrap();
    }
}

#[derive(Clone, Deref, ExtractResource)]
struct GameOfLifeImage(Handle<Image>);

#[derive(Clone, Deref, ExtractResource)]
struct GameOfLifeImageSecond(Handle<Image>);

struct GameOfLifeImageBindGroup(BindGroup);

fn prepare_params(
    sim_meta: Res<SimMeta>,
    render_queue: Res<RenderQueue>,
    time: Res<ExtractedTime>,
    mut sim_params: ResMut<SimParams>,
) {
    sim_params.time = time.seconds_since_startup;
    sim_params.delta = time.delta_time;
    sim_params.salt = rand::random::<u32>();

    let export = SimParamsExport {
        color: sim_params.color.to_array().map(|c| c as f32 / 255.0),
        blur_mask: sim_params.blur_mask.to_array().map(|c| c as f32 / 255.0),
        width: sim_params.width,
        height: sim_params.height,
        mode: sim_params.mode as u32,
        trail_weight: sim_params.trail_weight,
        decay_rate: sim_params.decay_rate,
        time: sim_params.time,
        delta: sim_params.delta,
        salt: sim_params.salt,
        move_speed: sim_params.move_speed,
        turn_speed: sim_params.turn_speed,
        sensor_angle_spacing: sim_params.sensor_angle_spacing,
        sensor_offset_distance: sim_params.sensor_offset_distance,
        sensor_size: sim_params.sensor_size,
    };

    render_queue.write_buffer(&sim_meta.params_buffer, 0, bytemuck::cast_slice(&[export]))
}

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<GameOfLifePipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    game_of_life_image: Res<GameOfLifeImage>,
    game_of_life_image_second: Res<GameOfLifeImageSecond>,
    sim_meta: Res<SimMeta>,
    render_device: Res<RenderDevice>,
) {
    let view = &gpu_images[&game_of_life_image.0];

    let view_second = &gpu_images[&game_of_life_image_second.0];

    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.texture_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&view_second.texture_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: sim_meta.agents_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: sim_meta.params_buffer.as_entire_binding(),
            },
        ],
    });
    commands.insert_resource(GameOfLifeImageBindGroup(bind_group));
}

// #[derive(Resource)]
pub struct GameOfLifePipeline {
    texture_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for GameOfLifePipeline {
    fn from_world(world: &mut World) -> Self {
        let texture_bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadWrite,
                                format: TextureFormat::Rgba8Unorm,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadWrite,
                                format: TextureFormat::Rgba8Unorm,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                // min_binding_size: BufferSize::new(std::mem::size_of::<
                                //     [Agent; NUM_AGENTS as usize],
                                // >(
                                // )
                                //     as u64),
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/game_of_life.wgsl");
        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![texture_bind_group_layout.clone()]),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![texture_bind_group_layout.clone()]),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        GameOfLifePipeline {
            texture_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

enum GameOfLifeState {
    Stopped,
    Init,
    Update,
}

struct GameOfLifeNode {
    state: GameOfLifeState,
}

impl Default for GameOfLifeNode {
    fn default() -> Self {
        Self {
            state: GameOfLifeState::Stopped,
        }
    }
}

impl render_graph::Node for GameOfLifeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GameOfLifePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let settings = world.resource::<SimSettings>();

        // if the corresponding pipeline has loaded, transition to the next stage SimState::Initialize => match self.state {
        match self.state {
            GameOfLifeState::Stopped => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                    self.state = GameOfLifeState::Init;
                }
            }
            GameOfLifeState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = GameOfLifeState::Update;
                }
            }
            GameOfLifeState::Update => {}
        }

        match settings.state {
            SimState::Initialize => self.state = GameOfLifeState::Init,
            _ => {}
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
        let pipeline = world.resource::<GameOfLifePipeline>();
        let settings = &world.resource::<SimSettings>();

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, texture_bind_group, &[]);

        // select the pipeline based on the current state
        match settings.state {
            SimState::Playing | SimState::Initialize => match self.state {
                GameOfLifeState::Stopped => {}
                GameOfLifeState::Init => {
                    let init_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.init_pipeline)
                        .unwrap();
                    pass.set_pipeline(init_pipeline);
                    pass.dispatch_workgroups(NUM_AGENTS / GAME_WORKGROUP_SIZE, 1, 1);
                }
                GameOfLifeState::Update => {
                    let update_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.update_pipeline)
                        .unwrap();
                    pass.set_pipeline(update_pipeline);
                    pass.dispatch_workgroups(NUM_AGENTS / GAME_WORKGROUP_SIZE, 1, 1);
                }
            },
            _ => {}
        }

        Ok(())
    }
}
