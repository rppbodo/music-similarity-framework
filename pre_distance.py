import os
import pylcs
import Levenshtein

import numpy as np

# =================================================================================================
# testing all transpositions for octave abstractions
# =================================================================================================
def transpose(sequence, t):
	return [(s + t) % 12 for s in sequence]

def compute_transposed_distances(dataset, distance_filename, tracks):
	if not os.path.isdir(os.path.join(dataset, "symbolic_domain")):
		os.mkdir(os.path.join(dataset, "symbolic_domain"))
	
	n = len(tracks)
	distance_matrix = np.ones((n, n)) * np.nan
	
	if "lcs" in distance_filename:
		for i in range(n):
			distance_matrix[i, i] = len(tracks[i].global_feat)
			
			sequence_i = "".join([chr(ord('A') + s) for s in tracks[i].global_feat])
			for j in range(i):
				max_len = float("-inf")
				for t in range(12):
					sequence_j = "".join([chr(ord('A') + s) for s in transpose(tracks[j].global_feat, t)])
					current_len = pylcs.lcs(sequence_i, sequence_j)
					if current_len > max_len:
						max_len = current_len
				distance_matrix[i, j] = max_len
				distance_matrix[j, i] = max_len
	
	elif "levenshtein" in distance_filename:
		for i in range(n):
			distance_matrix[i, i] = 0
			
			sequence_i = "".join([chr(ord('A') + s) for s in tracks[i].global_feat])
			for j in range(i):
				min_dist = float("inf")
				for t in range(12):
					sequence_j = "".join([chr(ord('A') + s) for s in transpose(tracks[j].global_feat, t)])
					current_dist = Levenshtein.distance(sequence_i, sequence_j)
					if current_dist < min_dist:
						min_dist = current_dist
				distance_matrix[i, j] = min_dist
				distance_matrix[j, i] = min_dist
	
	assert np.isfinite(distance_matrix).all() # just to be safe
	
	np.savez(distance_filename, distance_matrix)
	
	return distance_matrix

# =================================================================================================
# testing all transpositions for chromagrams
# =================================================================================================
def get_optimal_transposition_index(feat_a, feat_b):
	average_a = np.mean(feat_a, axis=0)
	average_b = np.mean(feat_b, axis=0)
	
	max_dot = float("-inf")
	oti = None
	for i in range(12):
		curr_dot = np.dot(average_a, np.roll(average_b, i))
		if curr_dot > max_dot:
			max_dot = curr_dot
			oti = i
	return oti

def compute_oti_matrix(dataset, oti_filename, tracks):
	if not os.path.isdir(os.path.join(dataset, "oti")):
		os.mkdir(os.path.join(dataset, "oti"))
	
	n = len(tracks)
	oti_matrix = np.ones((n, n)) * np.nan
	
	for i in range(n):
		oti_matrix[i, i] = 0
		for j in range(n):
			if i != j:
				oti_matrix[i, j] = get_optimal_transposition_index(tracks[i].local_feat, tracks[j].local_feat)
	
	assert np.isfinite(oti_matrix).all() # just to be safe
	
	np.savez(oti_filename, oti_matrix)
	
	return oti_matrix

