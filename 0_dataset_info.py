import sys

import dataset as ds

def main(dataset):
	classes, tracks = ds.load(dataset)
	print(dataset)
	print(len(classes), "classes")
	print(len(tracks), "tracks")
	print()
	
	for track in tracks:
		track.load()
		print(track)
	print()

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("error: invalid number of args")
		print()
		
		print("usage: python", sys.argv[0], "[dataset path]")
		print()
		
		sys.exit(1)
	
	dataset_param = sys.argv[1]
	main(dataset_param)

