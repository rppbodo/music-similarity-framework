import sys

import dataset as ds
import feature_aggregator as fa

def main(dataset, extractor, aggregator):
	classes, tracks = ds.load(dataset)
	print(len(classes), "classes")
	print(len(tracks), "tracks")
	print()

	print("aggregating features...")
	for track in tracks:
		track.global_feat = fa.feature_aggregator(track, extractor, aggregator)
		print(track.global_feat.shape)
	print()

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("error: invalid number of args")
		print()
		
		print("usage: python", sys.argv[0], "[dataset path] [extractor name] [aggregator name]")
		print()
		
		sys.exit(1)
	
	dataset_param = sys.argv[1]
	extractor_param = sys.argv[2]
	aggregator_param = sys.argv[3]
	main(dataset_param, extractor_param, aggregator_param)

