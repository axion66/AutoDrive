extends Node3D
class_name Roads

var gridmap = self.get_child(0)
var adjacency = {}

"""
# finds shortest path between 2 nodes of a graph using BFS
func bfs_shortest_path(graph, start, goal):
	# keep track of explored nodes
	var explored = []
	# keep track of all the paths to be checked
	var queue = [[start]]

	# return path if start is goal
	if start == goal:
		return [start]

	# keeps looping until all possible paths have been checked
	while queue:
		# pop the first path from the queue
		var path = queue.pop_at(0)
		# get the last node from the path
		
		print(explored)

		var node = path[-1]
		if node not in explored:
			var neighbours = graph[node]
			# go through all neighbour nodes, construct a new path and
			# push it into the queue
			for neighbour in neighbours:
				var new_path = path
				new_path.append(neighbour)
				queue.append(new_path)
				# return path if neighbour is goal
				if neighbour == goal:
					return new_path

			# mark node as explored
			explored.append(node)

	# in case there's no path between the 2 nodes
	return []
"""

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	for cell in gridmap.get_used_cells():
		adjacency[cell] = []
		if gridmap.get_used_cells().has(Vector3i(cell.x - 1, cell.y, cell.z)):
			adjacency[cell].append(Vector3i(cell.x - 1, cell.y, cell.z))
		if gridmap.get_used_cells().has(Vector3i(cell.x + 1, cell.y, cell.z)):
			adjacency[cell].append(Vector3i(cell.x + 1, cell.y, cell.z))
		if gridmap.get_used_cells().has(Vector3i(cell.x, cell.y, cell.z - 1)):
			adjacency[cell].append(Vector3i(cell.x, cell.y, cell.z - 1))
		if gridmap.get_used_cells().has(Vector3i(cell.x, cell.y, cell.z + 1)):
			adjacency[cell].append(Vector3i(cell.x, cell.y, cell.z + 1))

	pass # Replace with function body.

# Returns a random tile
func _random_tile():
	pass

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
