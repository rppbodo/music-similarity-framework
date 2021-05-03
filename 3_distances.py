import sys

import dataset as ds
import feature_aggregator as fa
import distance_calculator as dc

def main(dataset, extractor, aggregator, distance):
	classes, tracks = ds.load(dataset)
	print(len(classes), "classes")
	print(len(tracks), "tracks")
	print()
	
	print("loading global features...")
	for track in tracks:
		track.global_feat = fa.feature_aggregator(track, extractor, aggregator)
	print()
	
	print("computing distances...")
	for track in tracks:
		track.distances =  dc.distance_calculator(track.index, tracks, extractor, aggregator, distance)
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

