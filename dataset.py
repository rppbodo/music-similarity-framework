import sys
import csv

from track import *

def load_csv(path, csv_file):
	classes = []
	tracks = []
	
	csvfile = open(csv_file, "r")
	reader = csv.reader(csvfile)
	k = 0
	for row in reader:
		if not row[0] in classes:
			classes.append(row[0])
		basename, extension = os.path.splitext(row[1])
		tracks.append(Track(path, row[0], basename, extension, k))
		k += 1
	csvfile.close()
	
	return np.array(classes), np.array(tracks)

def load(path):
	csv_file = os.path.join(path, "tracks.csv")
	if os.path.isfile(csv_file):
		return load_csv(path, csv_file)
	
	raise ValueError("missing CSV file in " + path)

