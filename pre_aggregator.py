import os
import sys
import gzip
import joblib

import numpy as np

from sklearn.cluster import KMeans

import dataset as ds
import feature_extractor as fe

def compute_codebook(dataset, extractor, default):
	classes, tracks = ds.load(dataset)
	
	print("gathering local features...")
	matrix = None
	for track in tracks:
		track.local_feat = fe.feature_extractor(track, extractor)
		
		if matrix is None:
			matrix = track.local_feat
		else:
			matrix = np.concatenate((matrix, track.local_feat))
	
	print("creating codebook...")
	if default:
		kmeans = KMeans(n_clusters=8).fit(matrix)
		codebook_filename = os.path.join(dataset, "codebooks", extractor + "_codebook_kmeans_8_clusters.gz")		
	else:
		kmeans = KMeans(n_clusters=len(classes)).fit(matrix)
		codebook_filename = os.path.join(dataset, "codebooks", extractor + "_codebook_kmeans_" + str(len(classes)) + "_clusters.gz")
	
	if not os.path.isdir(os.path.join(dataset, "codebooks")):
		os.mkdir(os.path.join(dataset, "codebooks"))
	
	with gzip.GzipFile(codebook_filename, "wb", compresslevel=3) as fo:
		joblib.dump(kmeans, fo)
	
	return kmeans

