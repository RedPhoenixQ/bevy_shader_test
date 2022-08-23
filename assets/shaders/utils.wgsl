@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

@group(0) @binding(1)
var texture_second: texture_storage_2d<rgba8unorm, read_write>;

struct Agent {
    position: vec2<f32>,
    angle: f32
};

@group(0) @binding(2)
var<storage, read_write> agents: array<Agent>;

struct Params {
    color: vec4<f32>,
    blur_mask: vec4<f32>,
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
    };

@group(0) @binding(3)
var<uniform> params: Params;

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    state = state ^ state >> 16u; 
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

fn get_color(location: vec2<i32>, offset_x: i32, offset_y: i32) -> vec4<f32> {
    return textureLoad(texture, location + vec2<i32>(offset_x, offset_y));
}

fn average_neighbors(location: vec2<i32>) -> vec4<f32> {
    let neighbor_sum: vec4<f32> = get_color(location, -1, -1) +
           get_color(location, -1,  0) +
           get_color(location, -1,  1) +
           get_color(location,  0, -1) +
           get_color(location,  0,  1) +
           get_color(location,  1, -1) +
           get_color(location,  1,  0) +
           get_color(location,  1,  1);

    return neighbor_sum * vec4<f32>(0.125) * vec4<f32>(params.blur_mask.xyz, 1.0);
}

@compute @workgroup_size(16, 16, 1)
fn blur(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < u32(0) || id.x >= params.width || id.y < u32(0) || id.y >= params.height) {
        return;
    };
    let location = vec2<i32>(i32(id.x), i32(id.y));

    var sum: vec4<f32>;

    // for (var offsetX = -1; offsetX <= 1; offsetX ++) {
	// 	for (var offsetY = -1; offsetY <= 1; offsetY ++) {
	// 		let sampleX = min(i32(params.width) - 1, max(0, offsetX));
	// 		let sampleY = min(i32(params.height) - 1, max(0, offsetY));
            
	// 		sum += textureLoad(texture, vec2<i32>(sampleX, sampleY));
	// 	}
	// }

    let original_color = textureLoad(texture, location);
    let blurred_color = average_neighbors(location);
    let color = (original_color + (blurred_color - original_color) * params.trail_weight);

    storageBarrier();

    textureStore(texture, location, color);
}

@compute @workgroup_size(16, 16, 1)
fn decay(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < u32(0) || id.x >= params.width || id.y < u32(0) || id.y >= params.height) {
        return;
    };
    let location = vec2<i32>(i32(id.x), i32(id.y));

    let color = textureLoad(texture, location);

    let new_color = max(vec4<f32>(0.0), color - vec4<f32>(params.decay_rate * params.delta));

    // storageBarrier();

    textureStore(texture, location, new_color);
}

@compute @workgroup_size(16, 16, 1)
fn show_random(@builtin(global_invocation_id) id: vec3<u32>) {
      if (id.x < u32(0) || id.x >= params.width || id.y < u32(0) || id.y >= params.height) {
        return;
    };
    let location = vec2<i32>(i32(id.x), i32(id.y));

    let pixel_index = id.x * params.width + id.y;
    let random = randomFloat(pixel_index);
    textureStore(texture, location, vec4<f32>(random ,random,random, 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn color(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < u32(0) || id.x >= params.width || id.y < u32(0) || id.y >= params.height) {
        return;
    };
    let location = vec2<i32>(i32(id.x), i32(id.y));

    let map = textureLoad(texture_second, location);

    let color = vec4<f32>((params.color.xyz * map.xyz), 1.0);

    // storageBarrier();

    textureStore(texture, location, color);
}