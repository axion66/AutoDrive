extends AIController3D
class_name CarAIController

func get_obs_space():
	# may need overriding if the obs space is complex
	var obs = await get_obs()
	return {
		"obs": {
			"size": [len(obs["obs"])],
			"space": "box"
		},
	}
# Gets the observations of the car
func get_obs() -> Dictionary:
	var ball_pos = to_local(_player.global_transform.origin)
	var goal_local_pos = _player.to_local(_player.goal.global_transform.origin)
	var goal_distance = goal_local_pos.length()
	var goal_direction = goal_local_pos.normalized()
	var forward_direction = _player.global_transform.basis.z.normalized()
	
	var vector_observations: Array = [
		float(ball_pos.x),                     # Convert to float
		float(ball_pos.z),                     # Convert to float
		float(goal_local_pos.x),               # Convert to float
		float(goal_local_pos.z),               # Convert to float
		float(goal_distance),                   # Convert to float
		float(goal_direction.x),                # Convert to float
		float(goal_direction.z),                # Convert to float
		float(forward_direction.x),             # Convert to float
		float(forward_direction.z),             # Convert to float
		float(to_local(_player.linear_velocity).x) / 10,  # Convert to float and scale
		float(to_local(_player.linear_velocity).z) / 10,  # Convert to float and scale
		float(to_local(_player.angular_velocity).x) / 10, # Convert to float and scale
		float(to_local(_player.angular_velocity).z) / 10, # Convert to float and scale
		float(_player.steering) / deg_to_rad(_player.max_steer_angle)  # Convert to float
	]
	#vector_observations.append_array(_player.raycast_sensor.get_observation())	
		#somehow raycasting hates the goal
	'''
		dim:
			:orig_vec:14,
			:raycast:12,
			:camera:3072(32*32*3)
			
		:total:3098
	'''
	var pixel_observation = await _player.get_pixel_observation()
	vector_observations.append_array(pixel_observation)
	
	return {
		"obs": vector_observations,
	}

# Tells the reinforcement model the reward
func get_reward() -> float:
	return reward

# Tells the reinforcement model what the types of actions it can make (i.e. float, binary, int, etc.)
func get_action_space() -> Dictionary:
	return {
			"acceleration" : {
				"size": 1,
				"action_type": "continuous"
			},
			"steering" : {
				"size": 1, 
				"action_type": "continuous"
			},

		}
		
'''
# A function that updates every physics process, checks that the program doesn't last forever
func _physics_process(delta):
	if _player.episode_started:
		n_steps += 1
	if n_steps > reset_after:
		needs_reset = true
		done = true
'''
# Allows the reinforcement model to move the cars
func set_action(action) -> void:
	_player.requested_acceleration= clampf(action.acceleration[0],0,1.0)
	#print("angle: ",_player.requested_acceleration)
	'''
	
	
	if acceleration_value >= -1 && acceleration_value <= -0.05:
		_player.requested_acceleration = -1
	elif acceleration_value > -0.05 && acceleration_value <= 0.05:
		_player.requested_acceleration = 0
	else:  # acceleration_value > 0.6
		_player.requested_acceleration =  1

	#_player.requested_steering = action.steering[0] #clampf(action.steering[0], -1.0, 1.0)
	'''

	var steering_value = action.steering[0]

	if steering_value >= -0.33333 && steering_value <= 0.33333:
		_player.requested_steering = 0
	elif steering_value < -0.33333:
		_player.requested_steering = -1
	else:  # steering_value > 0.33333
		_player.requested_steering = 1

	#print("angle: ", action.steering[0] )
	#print("acceleration: ", _player.requested_acceleration)
func reset_episode():
	# Reset step counter and episode-related flags
	n_steps = 0
	needs_reset = false
	done = false
	reward = 0.0
	
	# Print debug information for tracking
	print("AI Controller: Episode reset")
