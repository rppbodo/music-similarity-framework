import sys

import dataset as ds
import feature_extractor as fe

def main(dataset, extractor):
	classes, tracks = ds.load(dataset)
	print(len(classes), "classes")
	print(len(tracks), "tracks")
	print()
	
	print("extracting features...")
	for track in tracks:
		track.local_feat = fe.feature_extractor(track, extractor)
		print(track.local_feat.shape)
	print()

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("error: invalid number of args")
		print()
		
		print("usage: python", sys.argv[0], "[dataset path] [extractor name]")
		print()
		
		sys.exit(1)
	
	dataset_param = sys.argv[1]
	extractor_param = sys.argv[2]
	main(dataset_param, extractor_param)

