import os
import sys

import numpy as np

from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine

def handle_directories(track, extractor_name, aggregator_name, distance_name):
	base_dir = os.path.join(track.path, "distances")
	if not os.path.exists(base_dir):
		print("creating", base_dir)
		os.makedirs(base_dir)
	
	class_dir = os.path.join(base_dir, track.class_)
	if not os.path.exists(class_dir):
		print("creating", class_dir)
		os.makedirs(class_dir)
	
	track_dir = os.path.join(class_dir, track.basename)
	if not os.path.exists(track_dir):
		print("creating", track_dir)
		os.makedirs(track_dir)
	
	return os.path.join(track_dir, extractor_name + "-" + aggregator_name + "-" + distance_name + ".npz")

def distance_calculator(i, tracks, extractor_name, aggregator_name, distance_name, **kwargs):
	filename = handle_directories(tracks[i], extractor_name, aggregator_name, distance_name)
	
	if not os.path.isfile(filename):
		print("computing", filename)
		distance = []
		thismodule = sys.modules[__name__]
		for j in range(len(tracks)):
			distance.append(getattr(thismodule, distance_name)(tracks[i], tracks[j]))
		distance = np.array(distance)
		np.savez(filename, distance)
		return distance
	else:
		print("loading", filename)
		distance_fp = np.load(filename)
		distance = distance_fp["arr_0"]
		distance_fp.close()
		return distance

# =================================================================================================
# spatial distances
# =================================================================================================
def euclidean_distance(track_a, track_b):
	return euclidean(track_a.global_feat, track_b.global_feat)

def manhattan_distance(track_a, track_b):
	return cityblock(track_a.global_feat, track_b.global_feat)

def chebyshev_distance(track_a, track_b):
	return chebyshev(track_a.global_feat, track_b.global_feat)

def cosine_distance(track_a, track_b):
	return cosine(track_a.global_feat, track_b.global_feat)

