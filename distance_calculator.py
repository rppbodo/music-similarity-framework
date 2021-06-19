import os
import sys
import pylcs
import Levenshtein

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
# distances between vectors
# =================================================================================================
def euclidean_distance(track_a, track_b):
	return euclidean(track_a.global_feat, track_b.global_feat)

def manhattan_distance(track_a, track_b):
	return cityblock(track_a.global_feat, track_b.global_feat)

def chebyshev_distance(track_a, track_b):
	return chebyshev(track_a.global_feat, track_b.global_feat)

def cosine_distance(track_a, track_b):
	return cosine(track_a.global_feat, track_b.global_feat)

# =================================================================================================
# distances between symbolic sequences
# =================================================================================================
def process_sequence(s):
	if type(s[0]) == np.str_:
		return "".join(s)
	
	# MIDI note numbers range from 0 to 127
	# so maximum interval = +127
	#    minimum interval = -127
	elif type(s[0]) == np.float64:
		temp = [f + 127 for f in s] # intervals need to be fixed to produce positive numbers
		return "".join([chr(int(f)) for f in temp]) # to be converted to single characters
	
	else:
		print(type(s[0]))
		raise TypeError("invalid type for sequence element")

def lcs_common(track_a, track_b, function):
	a = process_sequence(track_a.global_feat)
	b = process_sequence(track_b.global_feat)
	
	len_a = len(a)
	len_b = len(b)
	if len_a == 0 or len_b == 0:
		return 1
	
	lcs_len = pylcs.lcs(a, b)
	return 1 - lcs_len / function([len_a, len_b])

def lcs_max(track_a, track_b):
	return lcs_common(track_a, track_b, np.max)

def lcs_mean(track_a, track_b):
	return lcs_common(track_a, track_b, np.mean)

def lcs_min(track_a, track_b):
	return lcs_common(track_a, track_b, np.min)

def levenshtein_common(track_a, track_b, function):
	a = process_sequence(track_a.global_feat)
	b = process_sequence(track_b.global_feat)
	
	len_a = len(a)
	len_b = len(b)
	if len_a == 0 or len_b == 0:
		return 1
	
	levenshtein_dist = Levenshtein.distance(a, b)
	return levenshtein_dist / function([len_a, len_b])

def levenshtein_max(track_a, track_b):
	return levenshtein_common(track_a, track_b, np.max)

def levenshtein_mean(track_a, track_b):
	return levenshtein_common(track_a, track_b, np.mean)

def levenshtein_min(track_a, track_b):
	return levenshtein_common(track_a, track_b, np.min)

# =================================================================================================
# distances between Markov chains
# =================================================================================================
def circular_common(track_a, track_b, distance_function):
	u = track_a.global_feat.flatten()
	min_dist = float("inf")
	for shift in range(12):
		shifted_feat = np.zeros((12, 12))
		for i in range(12):
			for j in range(12):
				shifted_feat[(i + shift) % 12, (j + shift) % 12] = track_b.global_feat[i, j]
		v = shifted_feat.flatten()
		dist = distance_function(u, v)
		if dist < min_dist:
			min_dist = dist
	return min_dist

def circular_euclidean(track_a, track_b):
	return circular_common(track_a, track_b, euclidean)

def circular_manhattan(track_a, track_b):
	return circular_common(track_a, track_b, cityblock)

def circular_chebyshev(track_a, track_b):
	return circular_common(track_a, track_b, chebyshev)

def circular_cosine(track_a, track_b):
	return circular_common(track_a, track_b, cosine)

