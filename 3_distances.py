import os
import sys

import numpy as np

import dataset as ds
import feature_extractor as fe
import feature_aggregator as fa
import distance_calculator as dc
import pre_distance as pd

def main(dataset, extractor, aggregator, distance):
	classes, tracks = ds.load(dataset)
	print(len(classes), "classes")
	print(len(tracks), "tracks")
	print()
	
	kwargs = {}
	if distance.startswith("lcs_circular") or distance.startswith("levenshtein_circular"):
		print("loading global features...")
		for track in tracks:
			track.global_feat = fa.feature_aggregator(track, extractor, aggregator)
		print()
		
		key = distance.replace("_min", "").replace("_max", "").replace("_mean", "")
		basename = "_".join([extractor, aggregator, key]) + ".npz"
		distance_filename = os.path.join(dataset, "symbolic_domain", basename)
		
		if not os.path.isfile(distance_filename):
			distance_matrix = pd.compute_transposed_distances(dataset, distance_filename, tracks)
			print("distance matrix computed")
		else:
			distance_matrix_fp = np.load(distance_filename)
			distance_matrix = distance_matrix_fp["arr_0"]
			distance_matrix_fp.close()
			print("distance matrix loaded")
		
		kwargs[key] = distance_matrix
	
	elif distance.endswith("_oti"):
		oti_filename = os.path.join(dataset, "oti", extractor + ".npz")
		
		print("loading global features...")
		for track in tracks:
			track.local_feat = fe.feature_extractor(track, extractor)
			print(track.local_feat.shape)
		print()

		if not os.path.isfile(oti_filename):
			oti_matrix = pd.compute_oti_matrix(dataset, oti_filename, tracks)
			print("oti matrix computed")
		else:
			oti_matrix_fp = np.load(oti_filename)
			oti_matrix = oti_matrix_fp["arr_0"]
			oti_matrix_fp.close()
			print("oti matrix loaded")
		
		kwargs["oti_matrix"] = oti_matrix
		kwargs["aggregator"] = aggregator
	else:
		print("loading global features...")
		for track in tracks:
			track.global_feat = fa.feature_aggregator(track, extractor, aggregator)
		print()
	
	print("computing distances...")
	for track in tracks:
		track.distances =  dc.distance_calculator(track.index, tracks, extractor, aggregator, distance, **kwargs)
		print(track.distances.shape)
	print()

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print("error: invalid number of args")
		print()
		
		print("usage: python", sys.argv[0], "[dataset path] [extractor name] [aggregator name] [distance name]")
		print()
		
		sys.exit(1)
	
	dataset_param = sys.argv[1]
	extractor_param = sys.argv[2]
	aggregator_param = sys.argv[3]
	distance_param = sys.argv[4]
	main(dataset_param, extractor_param, aggregator_param, distance_param)

