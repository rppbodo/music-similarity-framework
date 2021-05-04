import os
import pickle
import numpy as np

def normalize(matrix):
	min_value = np.min(matrix)
	max_value = np.max(matrix)
	
	matrix = matrix - min_value
	matrix = matrix / (max_value - min_value)
	
	assert np.min(matrix) == 0.0
	assert np.max(matrix) == 1.0
	
	return matrix

def invert(matrix):
	matrix = 1.0 - matrix
	
	assert np.min(matrix) == 0.0
	assert np.max(matrix) == 1.0
	
	return matrix

def check_symmetry(matrix):
	n_positions = 0
	(n, m) = matrix.shape
	for i in range(n):
		for j in range(i):
			if matrix[i][j] != matrix[j][i]:
				n_positions += 1
	if n_positions > 0:
		print("this matrix is not symmetric in", n_positions, "positions")
	else:
		print("this matrix is symmetric")
	print()

def tracks_in_class_and_not(tracks, class_):
	tic = [] # tracks in class
	tnic = [] # tracks not in class
	for track in tracks:
		if track.class_ == class_:
			tic.append(track)
		else:
			tnic.append(track)
	return np.array(tic), np.array(tnic)

def intra_inter_class_similarity_ratio(tracks, class_, matrix):
	tic, tnic = tracks_in_class_and_not(tracks, class_)
	
	n_intra_values = len(tic) ** 2 - len(tic)
	n_inter_values = len(tnic) * len(tic)
	
	intra_similarity = 0
	for t1 in tic:
		for t2 in tic:
			if t1 != t2:
				intra_similarity += matrix[t1.index][t2.index]
	inter_similarity = 0
	for t1 in tic:
		for t2 in tnic:
			inter_similarity += matrix[t1.index][t2.index]
	
	return (intra_similarity / n_intra_values) / (inter_similarity / n_inter_values)

def compute_similarity_metrics(dataset, extractor, aggregator, distance, classes, tracks, similarity_matrix):
	metrics_filename = os.path.join(dataset, "results", extractor + "-" + aggregator + "-" + distance + "-similarity_metrics.pkl")
	if not os.path.isfile(metrics_filename):
		print("computing", metrics_filename)
		metrics = {}
		for class_ in classes:
			metrics[class_] = intra_inter_class_similarity_ratio(tracks, class_, similarity_matrix)
		metrics_fp = open(metrics_filename, "wb")
		pickle.dump(metrics, metrics_fp)
		metrics_fp.close()
		return metrics
	else:
		print("loading", metrics_filename)
		metrics_fp = open(metrics_filename, "rb")
		metrics = pickle.load(metrics_fp)
		metrics_fp.close()
		return metrics

def top_n(ranks, n):
	return np.sum([1 for rank in ranks if rank <= n])

def compute_cover_metrics(dataset, extractor, aggregator, distance, classes, tracks, similarity_matrix):
	metrics_filename = os.path.join(dataset, "results", extractor + "-" + aggregator + "-" + distance + "-cover_metrics.pkl")
	if not os.path.isfile(metrics_filename):
		print("computing", metrics_filename)
		
		# metrics computed with ranks from the first relevant items
		ranks = []
		for i in range(similarity_matrix.shape[0]):
			row = similarity_matrix[i]
			indexes = np.argsort(row)[::-1]
			ordered_tracks = tracks[indexes]
			assert tracks[i] == ordered_tracks[0]
			for j in range(1, len(row)):
				if tracks[i].class_ == ordered_tracks[j].class_:
					ranks.append(j)
					break
		ranks = np.array(ranks)
		
		mr = np.mean(ranks) # mean rank
		mrr = np.mean(1.0 / ranks) # mean reciprocal rank
		mdr = np.median(ranks) # median rank
		
		top_1 = top_n(ranks, 1)
		top_10 = top_n(ranks, 10)
		top_100 = top_n(ranks, 100)
		top_1000 = top_n(ranks, 1000)
		
		# metric computed with ranks from all relevant items
		apk = []
		for i in range(similarity_matrix.shape[0]):
			pk = []
			row = similarity_matrix[i]
			indexes = np.argsort(row)[::-1]
			ordered_tracks = tracks[indexes]
			assert tracks[i] == ordered_tracks[0]
			
			relevant_items = 0
			for j in range(1, similarity_matrix.shape[1]):
				if tracks[i].class_ == ordered_tracks[j].class_:
					relevant_items += 1
					rank = j
					pk.append(relevant_items/rank)
			apk.append(np.mean(pk))
		map_ = np.mean(apk) # mean average precision
		
		metrics = {}
		metrics["mr"] = mr
		metrics["mrr"] = mrr
		metrics["mdr"] = mdr
		metrics["map"] = map_
		metrics["top_1"] = top_1
		metrics["top_10"] = top_10
		metrics["top_100"] = top_100
		metrics["top_1000"] = top_1000
		
		metrics_fp = open(metrics_filename, "wb")
		pickle.dump(metrics, metrics_fp)
		metrics_fp.close()
		return metrics
	else:
		print("loading", metrics_filename)
		metrics_fp = open(metrics_filename, "rb")
		metrics = pickle.load(metrics_fp)
		metrics_fp.close()
		return metrics

