import numpy as np

def compare(fitness_a, fitness_b):
		# if a dominates b, return will be 1, -1 if the contrary
		# if there is no dominance, 0 will be returned
		if fitness_a[1] - fitness_b[1] == 0.0:
			if fitness_a[0] < fitness_b[0]:
				return 1
			return -1
		try:
			m = (fitness_a[0] - fitness_b[0]) / (fitness_a[1] - fitness_b[1])
		except ZeroDivisionError:
			return 0
		if m > 0:
			if fitness_a[0] < fitness_b[0]:
				return 1
			return -1
		return 0

def NSGA2(fitness_array:np.ndarray):
	n_indiv = len(fitness_array)
	levels = np.zeros(n_indiv)
	dominates = [[] for _ in range(n_indiv)]
	is_dominated_by = {}

	for i in range(n_indiv):
		is_dominated_by[i] = 0

	for idx_first_point, first_point in enumerate(fitness_array):
		for idx_second_point in range(idx_first_point+1, n_indiv):
			second_point = fitness_array[idx_second_point]
			m = compare(first_point, second_point)

			if m == 1: #first_point dominates
				is_dominated_by[idx_second_point] +=1
				dominates[idx_first_point].append(idx_second_point)
			elif m == -1:
				is_dominated_by[idx_first_point] +=1
				dominates[idx_second_point].append(idx_first_point)

	curr_level = 0
	while len(is_dominated_by) > 0:
		non_dominated_indices = [
			idx for idx, dominance_count in is_dominated_by.items() if dominance_count == 0
		]

		for idx in non_dominated_indices:
			for dominated_idx in dominates[idx]:
				is_dominated_by[dominated_idx] -= 1

			levels[idx] = curr_level
			del is_dominated_by[idx]
		curr_level+=1
			
	return levels
