import sys

import dataset as ds
import distance_calculator as dc
import metric_calculator as mc
import utils as u

def print_metric_type_options():
	print("[metric type]")
	str_1 = "--similarity or -s"
	str_2 = "Compute intra-inter class similarity ratios"
	str_3 = "--cover or -c"
	str_4 = "Compute cover song identification metrics"
	m = len(str_1)
	n = len(str_3)
	if m > n:
		print(str_1, "\t", str_2)
		print(str_3, (m - n - 1) * " ", "\t", str_4)
	else:
		print(str_1, (n - m - 1) * " ", "\t", str_2)
		print(str_3, "\t", str_4)

def main(dataset, extractor, aggregator, distance, metric_type):
	classes, tracks = ds.load(dataset)
	print(len(classes), "classes")
	print(len(tracks), "tracks")
	print()
	
	print("loading distancies...")
	for track in tracks:
		track.distances = dc.distance_calculator(track.index, tracks, extractor, aggregator, distance)
	print()
	
	similarity_matrix = mc.metric_calculator(dataset, tracks, extractor, aggregator, distance)
	print(similarity_matrix.shape)
	print()
	
	if metric_type_param == "--similarity" or metric_type_param == "-s":
		metrics = u.compute_similarity_metrics(dataset, extractor, aggregator, distance, classes, tracks, similarity_matrix)
	elif metric_type_param == "--cover" or metric_type_param == "-c":
		metrics = u.compute_cover_metrics(dataset, extractor, aggregator, distance, classes, tracks, similarity_matrix)
	print(metrics)
	print()

if __name__ == "__main__":
	if len(sys.argv) != 6:
		print("error: invalid number of args")
		print()
		
		print("usage: python", sys.argv[0], "[dataset path] [extractor name] [aggregator name] [distance name] [metric type]")
		print()
		
		print_metric_type_options()
		print()
		
		sys.exit(1)
	
	dataset_param = sys.argv[1]
	extractor_param = sys.argv[2]
	aggregator_param = sys.argv[3]
	distance_param = sys.argv[4]
	metric_type_param = sys.argv[5]
	assert metric_type_param == "--similarity" or metric_type_param == "-s" or metric_type_param == "--cover" or metric_type_param == "-c"
	main(dataset_param, extractor_param, aggregator_param, distance_param, metric_type_param)

