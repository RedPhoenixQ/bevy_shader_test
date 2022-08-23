@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

@group(0) @binding(1)
var texture_second: texture_storage_2d<rgba8unorm, read_write>;

struct Agent {
    position: vec2<f32>,
    angle: f32,
};

@group(0) @binding(2)
var<storage, read_write> agents: array<Agent>;

struct Params {
    color: vec4<f32>,
    blur_mask: vec4<f32>,
    width: i32,
    height: i32,
    mode: i32,
    trail_weight: f32,
    decay_rate: f32,
    time: f32,
    delta: f32,
    salt: u32,
    move_speed: f32,
    turn_speed: f32,
    sensor_angle_spacing: f32,
    sensor_offset_distance: f32,
    sensor_size: i32,
};

@group(0) @binding(3)
var<uniform> params: Params;

let pi = 3.14159265359;

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

@compute @workgroup_size(512, 1, 1)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    // let random = hash(id.x * params.width + id.x + hash(id.x + u32(params.time * 100000.123))) * u32(params.seed * params.delta);
    let random = hash(id.x * u32(params.width) * params.salt + u32(params.time * 100000.0));

    let center = vec2<f32>(f32(params.width) / 2.0, f32(params.height) / 2.0);

    let agent_id = id.x;
    let agent = &agents[agent_id];

    // (*agent).position = vec2<f32>((randomFloat(id.x * random * u32((random + id.x) ^ 2123u) + id.x) * f32(params.width)), (randomFloat(id.x * u32(params.delta * params.seed) * random + id.x)  * f32(params.height)));


    if (params.mode == 0) {
        (*agent).position = center;
        (*agent).angle = randomFloat(random) * 2.0 * pi;
    } else if (params.mode == 1) {
        let theta = randomFloat(random) * 2.0 * pi;
        let seed = hash(random + params.salt);
        let r = f32(params.height) * 0.25 * sqrt(randomFloat(seed));

        (*agent).position.x = center.x + r * sin(theta);
        (*agent).position.y = center.y + r * cos(theta);

        (*agent).angle = theta + pi;
    } else if (params.mode == 2) {
        (*agent).position = vec2<f32>((randomFloat(id.x * random + id.x) * f32(params.width)), (randomFloat(id.x * u32(params.delta) * params.salt * random + id.x)  * f32(params.height)));
            
        (*agent).angle = randomFloat(random) * 2.0 * pi;
    }


    let location = vec2<i32>(i32((*agent).position.x), i32((*agent).position.y));
    textureStore(texture, location, vec4<f32>(params.color.xyz, 1.0));
}

fn sense(agent: Agent, sensor_angle_spacing: f32) -> f32 {
    let sensor_angle = agent.angle + sensor_angle_spacing;
    let sensorDir = vec2<f32>(cos(sensor_angle), sin(sensor_angle));

	let sensorPos = agent.position + sensorDir * vec2<f32>(params.sensor_offset_distance);
	let sensorCentreX = i32(sensorPos.x);
	let sensorCentreY = i32(sensorPos.y);

    var sum: f32;

    for (var offsetX = params.sensor_size * -1; offsetX <= params.sensor_size; offsetX ++) {
		for (var offsetY = params.sensor_size * -1; offsetY <= params.sensor_size; offsetY ++) {
			let sampleX = min(params.width - 1, max(0, sensorCentreX + offsetX));
			let sampleY = min(params.height - 1, max(0, sensorCentreY + offsetY));
            let pixel_vec = textureLoad(texture, vec2<i32>(i32(sampleX),i32(sampleY)));
			sum += pixel_vec.w;
		}
	}

    return sum;

}

@compute @workgroup_size(512, 1, 1)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    // let random = hash(id.x * params.width + id.x + hash(id.x + u32(params.time * params.delta * 100000.123)) * u32(params.seed * 1200.90));

    let agent_id = id.x;
    let agent = &agents[agent_id];
    let location = vec2<i32>(i32((*agent).position.x), i32((*agent).position.y));


    let random = hash(id.x * u32(params.width) + hash(u32((*agent).position.x * (*agent).position.y)) + u32(params.time * 100000.0));

    let random_steer_strength = randomFloat(random);
	let turn_speed = params.turn_speed * 2.0 * pi;

	// Steer based on sensory data
	let sensorAngleRad = params.sensor_angle_spacing * (pi / 180.0);
	let weightForward = sense((*agent), 0.0);
	let weightLeft = sense((*agent), sensorAngleRad);
	let weightRight = sense((*agent), -sensorAngleRad);

    // Continue in same direction
	if (weightForward > weightLeft && weightForward > weightRight) {
		(*agent).angle += 0.0;
	}
	else if (weightForward < weightLeft && weightForward < weightRight) {
		(*agent).angle += (random_steer_strength - 0.5) * 2.0 * turn_speed * params.delta;
	}
	// Turn right
	else if (weightRight > weightLeft) {
		(*agent).angle -= random_steer_strength * turn_speed * params.delta;
	}
	// Turn left
	else if (weightLeft > weightRight) {
		(*agent).angle += random_steer_strength * turn_speed * params.delta;
	}

    let direction = vec2<f32>(cos((*agent).angle), sin((*agent).angle));
    // Movement to new position
    var new_pos = (*agent).position + direction * params.move_speed * params.delta;

    if (new_pos.x < 0.0 || new_pos.x >= f32(params.width) || new_pos.y < 0.0 || new_pos.y >= f32(params.height)) {
        (*agent).angle = randomFloat(random) * 2.0 * pi;
        let new_direction = vec2<f32>(cos((*agent).angle), sin((*agent).angle));
        new_pos = (*agent).position + new_direction * params.move_speed * params.delta;

        // new_pos.x = min(f32(params.width) - 1.0, max(0.0, new_pos.x));
        // new_pos.y = min(f32(params.height) - 1.0, max(0.0, new_pos.y));
    };

    (*agent).position = new_pos;

    // storageBarrier();

    textureStore(texture, location, vec4<f32>(params.color.xyz, 1.0));
}