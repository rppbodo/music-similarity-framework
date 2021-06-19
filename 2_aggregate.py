import os
import sys
import gzip
import joblib

import dataset as ds
import feature_aggregator as fa
import pre_aggregator as pa

def main(dataset, extractor, aggregator):
	classes, tracks = ds.load(dataset)
	print(len(classes), "classes")
	print(len(tracks), "tracks")
	print()
	
	kwargs = {}
	if aggregator.startswith("vector_quantization"):
		if aggregator.endswith("default"):
			codebook_filename = os.path.join(dataset, "codebooks", extractor + "_codebook_kmeans_8_clusters.gz")		
			kwargs["n_clusters"] = 8
			
			if not os.path.isfile(codebook_filename):
				pa.compute_codebook(dataset, extractor, True)
		else:
			codebook_filename = os.path.join(dataset, "codebooks", extractor + "_codebook_kmeans_" + str(len(classes)) + "_clusters.gz")
			kwargs["n_clusters"] = len(classes)
			
			if not os.path.isfile(codebook_filename):
				pa.compute_codebook(dataset, extractor, False)
		
		with gzip.GzipFile(codebook_filename, "rb") as fo:
			codebook = joblib.load(fo)
			print("codebook loaded:", codebook)
			kwargs["codebook"] = codebook
	
	print("aggregating features...")
	for track in tracks:
		track.global_feat = fa.feature_aggregator(track, extractor, aggregator, **kwargs)
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

