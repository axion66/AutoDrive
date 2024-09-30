extends VehicleBody3D
class_name Car

# Gets the sensors, goal, and path
@export var playing_area_x_size: float = 500
@export var playing_area_z_size: float = 500
@export var acceleration: float = 0.2
@export var max_steer_angle: float = 60
@onready var max_velocity = 4
@onready var ai_controller: AIController3D = $AIController3D
@onready var raycast_sensor: RayCastSensor3D = $RayCastSensor3D
@onready var goal: MeshInstance3D = $"../Goal"
@onready var road: Node3D = $"../Roads"
var gridmap: GridMap
var episode_started: bool = true
var episode_timer: float = 0
var max_episode_time: float = 100
var rng = RandomNumberGenerator.new()
var requested_acceleration: float
var requested_steering: float
var _initial_transform: Transform3D
var times_restarted: int
var _smallest_distance_to_goal: float = 1.8
var _max_goal_dist: float = 1
var episode_ended_unsuccessfully_reward: float = -0.5
var _rear_lights: Array[MeshInstance3D]
var starting_distance = 0
'''
	For camera!
	ai_controller
		-viewport
			--camera3d
'''
var viewport: SubViewport
var viewport_texture: ViewportTexture
var observation_camera: Camera3D

'''
	for "stuck (glitch)" issue
'''
var previous_position: Vector3 = Vector3.ZERO
var position_static_frames: int = 0
var max_static_frames: int = 3000  # Reset if position hasn't changed for 3000 frames


@export var braking_material: StandardMaterial3D
@export var reversing_material: StandardMaterial3D
@onready var front1 = $FrontWheel
@onready var front2 = $FrontWheel2
@onready var back1 = $BackWheel
@onready var back2 = $BackWheel2
var road_cells: Array
var ground: StaticBody3D

# gets position on the road
func get_random_road_position() -> Vector3:
	# Define the three specific coordinates
	var positions = [
		Vector3(12.575, 1.3, -14.639),
	]

	var random_position = positions[randi() % positions.size()]
	
	return random_position

# apply get_random_road_position to the car
func reset_car_position():
	var idx = rng.randi_range(0, len(gridmap.get_used_cells()) - 1)

	var new_position = road.to_global(gridmap.map_to_local(gridmap.get_used_cells()[0]))
	
	global_transform.origin = get_random_road_position()
	
	# Randomize rotation (0, 90, 180, or 270 degrees)
	rotation_degrees.y = randi() % 4 * 90
	
# apply get_random_road_position to the destination
func reset_goal_position():
	var new_goal_position = get_random_road_position()
	
	while new_goal_position.distance_to(global_transform.origin) < 10: 
		new_goal_position = get_random_road_position()
	
	goal.global_transform.origin = new_goal_position
	
# normalize velocity based on max_velocity
func get_normalized_velocity():
	return linear_velocity.normalized() * (linear_velocity.length() / max_velocity)


# ----------------- MAIN --------------------------

# This function is called at the beginning of the execution
func _ready():

	ground = get_parent().get_node("Ground")
	gridmap = road.get_child(0)
	road_cells = gridmap.get_used_cells()
	#reset_goal_position()
	
	ai_controller.init(self)
	_initial_transform = transform
		
	_rear_lights.resize(2)
	_rear_lights[0] = $"car_base/Rear-light" as MeshInstance3D
	_rear_lights[1] = $"car_base/Rear-light_001" as MeshInstance3D
	
	_max_goal_dist = (
		Vector2(
			playing_area_x_size,
			playing_area_z_size
		).length()
	)
	
	
	
	'''
		Camera setup
	'''
	viewport = SubViewport.new()
	viewport.size = Vector2i(32, 32)  # Adjust size as needed
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(viewport)

	# Add observation camera to the viewport
	observation_camera = Camera3D.new()
	viewport.add_child(observation_camera)
	observation_camera.current = true

	# Position the camera relative to the car
	observation_camera.transform = transform
	observation_camera.translate(Vector3(0, 2, -5))  # Adjust these values as needed
	observation_camera.look_at(global_position, Vector3.UP)
	starting_distance = _get_current_distance_to_goal()
	
# reset the episode. called every episode
func reset():
	episode_started = true
	times_restarted += 1
	position_static_frames = 0
	#reset_goal_position()
	
	transform = _initial_transform
	episode_timer = 0
	linear_velocity = Vector3.ZERO
	angular_velocity = Vector3.ZERO
	engine_force = 0
	steering = 0
	_smallest_distance_to_goal = _get_current_distance_to_goal()
	starting_distance = _get_current_distance_to_goal()
	#print("Episode reset complete")
	#reset_car_position()
	#reset_goal_position()
# check if the car is not on the road or not
func _is_on_road() -> bool:

	
	if (front1.get_contact_body() is Ground || front2.get_contact_body() is Ground \
		|| back1.get_contact_body() is Ground || back2.get_contact_body() is Ground):
		return true
	return false

func get_pixel_observation():
	# Wait for the next frame to ensure the viewport is updated
	await get_tree().process_frame
	
	# Render the viewport to a texture
	viewport_texture = viewport.get_texture()
	var image = viewport_texture.get_image()
	
	# Convert the image to a format suitable for the AI agent
	image.convert(Image.FORMAT_RGB8)
	var pixel_data = image.get_data()
	

	# Convert pixel data to a format expected by your AI framework
	# This might involve normalization, reshaping, etc.
	# Example: Convert to an array of normalized RGB values
	var observation = []
	for i in range(0, pixel_data.size(), 3):
		var r = float(pixel_data[i]) / 255.0
		var g = float(pixel_data[i+1]) / 255.0
		var b = float(pixel_data[i+2]) / 255.0
		observation.append_array([r,g,b])
	
	return observation
	
# called every milliseconds(100). defines physics of the car
func _physics_process(delta):
	episode_timer += 0.001 
	_update_reward()
		
	


	
	var current_position = global_transform.origin
	if current_position.distance_to(previous_position) < 0.001:
		position_static_frames += 1
	else:
		position_static_frames = 0

	previous_position = current_position	

	# Reset if the car has been static for too long
	if position_static_frames >= max_static_frames:
		print("Car static for too long, resetting...")
		_end_episode(-300)  # Penalize for being static
		reset()


	# if a human is controlling, allow for wasd controls. else, allow for the ai model to control the car
	#print("ai-val",ai_controller._player.requested_acceleration)
	engine_force = (ai_controller._player.requested_acceleration) * acceleration*100
	steering = deg_to_rad(ai_controller._player.requested_steering * max_steer_angle)
	#print(engine_force)
	#print(steering)
	#print("force:",engine_force)
	#print("steering:",steering)

	
	
	#_update_rear_lights() for changing the color of the rear lights.
	#_reset_on_out_of_bounds()
	#_reset_on_turned_over()
	#_reset_on_went_away_from_goal()
	#_end_episode_on_goal_reached()
	
	#observation_camera.global_transform = global_transform
	#observation_camera.translate_object_local(Vector3(0, 2, -5))  # Adjust these values as needed
	#observation_camera.look_at(global_position, Vector3.UP)
	

# ------------ REAR LIGHTS ---------------

# changes the rear lights so that they become red when the car is going backwards
func _update_rear_lights():
	var velocity := get_normalized_velocity_in_player_reference().z
	
	set_rear_light_material(null)
	
	var brake_or_reverse_requested: bool
	if (ai_controller.heuristic != "human"):
		brake_or_reverse_requested = requested_acceleration < 0
	else:
		brake_or_reverse_requested = Input.is_action_pressed("move_backward")
	
	if velocity >= 0:
		if brake_or_reverse_requested:
			set_rear_light_material(braking_material)
	elif velocity <= -0.015:
		set_rear_light_material(reversing_material)


# changes the rear light material
func set_rear_light_material(material: StandardMaterial3D):
	_rear_lights[0].set_surface_override_material(0, material)
	_rear_lights[1].set_surface_override_material(0, material)

# -------------------- RESETS ---------------

# CONSIDER CHANGING IN THE FUTURE (_smallest_distance_to_goal < x, x is too small)
# If the agent was near the goal but has since moved away,
# end the episode with a negative reward	
func _reset_on_went_away_from_goal():
	var goal_dist = _get_current_distance_to_goal()
	if _smallest_distance_to_goal < 1.5 and goal_dist > _smallest_distance_to_goal + 3.5:
		_end_episode(episode_ended_unsuccessfully_reward)

# reset and penalize if: car is turned over
func _reset_on_turned_over():
	if global_transform.basis.y.dot(Vector3.UP) < 0.6:
		_end_episode(episode_ended_unsuccessfully_reward)

# reset and penalize if: car is outside of the map
func _reset_on_out_of_bounds():
	if (abs(position.y) > 10 or abs(position.x) > 100 or abs(position.z) > 100):
		_end_episode(episode_ended_unsuccessfully_reward)


# ----------------------- REWARD FUNCTIONS ------------------------

# ends the current episode (right before a reset) and rewards the ai model accordingly
func _end_episode(final_reward: float = 0):
	ai_controller.reward += final_reward - episode_timer
	ai_controller.needs_reset = true
	ai_controller.done = true
	print("Culmulative Rewards: ", ai_controller.reward)
	

# If the goal condition is reached, reset and provide a (good) reward based on:
# how quickly the goal was reached,
# and distance from the goal position
func _end_episode_on_goal_reached():
	var goal_dist = _get_current_distance_to_goal()
	if _is_goal_reached(goal_dist):
		var parked_succesfully_reward: float = (
			10
			- (float(ai_controller.n_steps) / ai_controller.reset_after) * 2
			- (goal_dist / _max_goal_dist)
		)
		_end_episode(parked_succesfully_reward)


# Constantly rewards (per tick) in an episode	
func _update_reward():
	if times_restarted == 0:
		return
	
	var goal_dist = _get_current_distance_to_goal()
	if (goal_dist < 7):
		ai_controller.reward += 0.3
	elif (goal_dist < 14):
		ai_controller.reward += 0.25
	elif (goal_dist < 21):
		ai_controller.reward += 0.2
	elif (goal_dist < 28):
		ai_controller.reward += 0.15
	elif (goal_dist < starting_distance - 14):
		ai_controller.reward += 0.1
	elif (goal_dist < starting_distance - 6):
		ai_controller.reward += 0.05
	elif (goal_dist < starting_distance):
		ai_controller.reward += 0.01
	
	# Encourage shorter paths
	if goal_dist > starting_distance:
		ai_controller.reward -= 0.2
		
	if _is_goal_reached(_get_current_distance_to_goal()):
		#print("goal reached!")
		_end_episode(40)
		reset()
		
	if _is_on_road() or global_transform.origin.y < -5:
		#print("Fallen Reward")
		_end_episode(-30)
		reset()
	'''
	if episode_timer > max_episode_time:
		print("Time exceed reward")
		_end_episode(-30)
		reset()
	'''
	

# ------------------------------- HELPER FUNCTIONS ---------------------

# checks if the car reached the goal
func _is_goal_reached(current_goal_dist: float) -> bool:
	return (current_goal_dist < 1.3)

# Gets the current distance from the car to the goal
# ASSUMES THE MAP IS FLAT (GOAL.Y equals CAR.Y)
func _get_current_distance_to_goal() -> float:

	return goal.position.distance_to(global_position)

# gets the normalized velocity of the car from its frame of refrence
func get_normalized_velocity_in_player_reference() -> Vector3:
	return (
		global_transform.basis.inverse() *
		get_normalized_velocity()
		)

# if the car touches an obstacle, StaticCar, then its penalized and reset
func _on_body_entered(body: PhysicsBody3D):
	#if body is StaticCar:
	_end_episode(episode_ended_unsuccessfully_reward)
	
