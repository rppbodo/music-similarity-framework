import os

import numpy as np

import utils as u

def handle_directories(dataset, extractor_name, aggregator_name, distance_name):
	results_dir = os.path.join(dataset, "results")
	if not os.path.exists(results_dir):
		print("creating", results_dir)
		os.makedirs(results_dir)
	
	return os.path.join(results_dir, extractor_name + "-" + aggregator_name + "-" + distance_name + ".npz")

def metric_calculator(dataset, tracks, extractor_name, aggregator_name, distance_name):
	filename = handle_directories(dataset, extractor_name, aggregator_name, distance_name)
	
	if not os.path.isfile(filename):
		print("computing", filename)
		distance_matrix = None
		for track in tracks:
			if distance_matrix is None:
				distance_matrix = track.distances
			else:
				distance_matrix = np.vstack((distance_matrix, track.distances))
		
		assert np.isfinite(distance_matrix).all()
		
		distance_matrix = u.normalize(distance_matrix)
		similarity_matrix = u.invert(distance_matrix)
		u.check_symmetry(similarity_matrix)
		np.savez(filename, similarity_matrix)
		return similarity_matrix
	else:
		print("loading", filename)
		similarity_fp = np.load(filename)
		similarity_matrix = similarity_fp["arr_0"]
		similarity_fp.close()
		return similarity_matrix

