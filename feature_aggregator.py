import os
import sys
import gzip
import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

import feature_extractor as fe

def handle_directories(track, extractor_name, aggregator_name):
	base_dir = os.path.join(track.path, "global_features")
	if not os.path.exists(base_dir):
		print("creating", base_dir)
		os.makedirs(base_dir)
	
	class_dir = os.path.join(base_dir, track.class_)
	if not os.path.exists(class_dir):
		print("creating", class_dir)
		os.makedirs(class_dir)
	
	track_dir = os.path.join(class_dir, track.basename)
	if not os.path.exists(track_dir):
		print("creating", track_dir)
		os.makedirs(track_dir)
	
	return os.path.join(track_dir, extractor_name + "-" + aggregator_name + ".gz")

def feature_aggregator(track, extractor_name, aggregator_name, **kwargs):
	filename = handle_directories(track, extractor_name, aggregator_name)
	
	if aggregator_name == "bypass":
		return fe.feature_extractor(track, extractor_name).flatten()
	
	if not os.path.isfile(filename):
		print("computing", filename)
		track.local_feat = fe.feature_extractor(track, extractor_name)
		thismodule = sys.modules[__name__]
		global_feat = getattr(thismodule, aggregator_name)(track, **kwargs)
		track.unload()
		with gzip.GzipFile(filename, "wb", compresslevel=3) as fo:
			joblib.dump(global_feat, fo)
		return global_feat
	else:
		print("loading", filename)
		with gzip.GzipFile(filename, "rb") as fo:
			global_feat = joblib.load(fo)
			return global_feat

# ================================================================================================
# single gaussian methods
# ================================================================================================
def basic_stats_1(track):
	mean_values = np.mean(track.local_feat, axis=0)
	std_values = np.std(track.local_feat, axis=0)
	global_feat = np.concatenate((mean_values, std_values))
	assert len(global_feat) == 2 * track.local_feat.shape[1]
	return global_feat

def basic_stats_2(track):
	mean_values = np.mean(track.local_feat, axis=0)
	std_values = np.std(track.local_feat, axis=0)
	min_values = np.min(track.local_feat, axis=0)
	max_values = np.max(track.local_feat, axis=0)
	global_feat = np.concatenate((mean_values, std_values, min_values, max_values))
	assert len(global_feat) == 4 * track.local_feat.shape[1]
	return global_feat

def diff_stats_1(track):
	mean_values = np.mean(track.local_feat, axis=0)
	std_values = np.std(track.local_feat, axis=0)
	diff_mean_values = np.mean(np.diff(track.local_feat, axis=0), axis=0)
	diff_std_values = np.std(np.diff(track.local_feat, axis=0), axis=0)
	global_feat = np.concatenate((mean_values, std_values, diff_mean_values, diff_std_values))
	assert len(global_feat) == 4 * track.local_feat.shape[1]
	return global_feat

def diff_stats_2(track):
	mean_values = np.mean(track.local_feat, axis=0)
	std_values = np.std(track.local_feat, axis=0)
	min_values = np.min(track.local_feat, axis=0)
	max_values = np.max(track.local_feat, axis=0)
	diff_mean_values = np.mean(np.diff(track.local_feat, axis=0), axis=0)
	diff_std_values = np.std(np.diff(track.local_feat, axis=0), axis=0)
	diff_min_values = np.min(np.diff(track.local_feat, axis=0), axis=0)
	diff_max_values = np.max(np.diff(track.local_feat, axis=0), axis=0)
	global_feat = np.concatenate((mean_values, std_values, min_values, max_values, diff_mean_values, diff_std_values, diff_min_values, diff_max_values))
	assert len(global_feat) == 8 * track.local_feat.shape[1]
	return global_feat

def statistical_summarization(track):
	n_features = track.local_feat.shape[1]
	means = np.mean(track.local_feat, axis=0)
	
	covariance_matrix = np.cov(track.local_feat.T) # numpy.cov expects each row representing a variable, and each column a single observation of all those variables.
	variances = np.diagonal(covariance_matrix)
	covariances = []
	for i in range(1, n_features):
		covariances.extend(np.diagonal(covariance_matrix, offset=i))
	
	global_feat = np.concatenate((np.concatenate((means, variances)), covariances))
	assert len(global_feat) == 2 * n_features + (n_features * (n_features - 1)) / 2
	return global_feat

# ================================================================================================
# gaussian mixture model
# ================================================================================================
def gaussian_mixture_model_default(track):
	n_components = 8
	n_features = track.local_feat.shape[1]
	
	gmm = GaussianMixture(n_components=n_components, max_iter=200)
	gmm.fit(track.local_feat)
	global_feat = gmm.weights_	
	for i in range(n_components):
		means = gmm.means_[i]
		variances = np.diagonal(gmm.covariances_[i])
		covariances = []
		for j in range(1, n_features):
			covariances.extend(np.diagonal(gmm.covariances_[i], offset=j))
		global_feat = np.concatenate((np.concatenate((np.concatenate((global_feat, means)), variances)), covariances))
	
	assert len(global_feat) == n_components * (2 * n_features + (n_features * (n_features - 1)) / 2) + n_components
	return global_feat

# ================================================================================================
# vector quantization
# ================================================================================================
def relative_frequencies(feature, codebook, n_clusters):
	f = np.zeros(n_clusters)
	(n, m) = feature.shape
	for col in range(m):
		cluster = codebook.predict(feature[:,col].reshape(1, -1))
		f[cluster] += 1
	total = np.sum(f)
	for i in range(len(f)):
		f[i] /= total
	return f

def vector_quantization(track, **kwargs):
	codebook = kwargs["codebook"]
	n_clusters = kwargs["n_clusters"]
	return relative_frequencies(track.local_feat.T, codebook, n_clusters)

def vector_quantization_default(track, **kwargs):
	codebook = kwargs["codebook"]
	n_clusters = kwargs["n_clusters"]
	return relative_frequencies(track.local_feat.T, codebook, n_clusters)

# ================================================================================================
# aggregators for symbolic sequences
# ================================================================================================
def octave_abstraction(track):
	return np.array([int(p) % 12 for p in track.local_feat])

def interval_abstraction(track):
	length = len(track.local_feat)
	if length < 2:
		return []
	
	diff = np.zeros(length - 1)
	for i in range(1, length):
		diff[i-1] = track.local_feat[i] - track.local_feat[i-1]
	return diff

def pitch_contour_3_levels(track):
	pitch_contour = []
	for i in range(1, len(track.local_feat)):
		if track.local_feat[i] > track.local_feat[i-1]:
			pitch_contour.append("u")
		elif track.local_feat[i] < track.local_feat[i-1]:
			pitch_contour.append("d")
		else:
			pitch_contour.append("r")
	return np.array(pitch_contour)

def pitch_contour_5_levels(track):
	pitch_contour = []
	for i in range(2, len(track.local_feat)):
		if track.local_feat[i] > track.local_feat[i-2]:
			pitch_contour.append("U")
		elif track.local_feat[i] < track.local_feat[i-2]:
			pitch_contour.append("D")
		elif track.local_feat[i] > track.local_feat[i-1]:
			pitch_contour.append("u")
		elif track.local_feat[i] < track.local_feat[i-1]:
			pitch_contour.append("d")
		else:
			pitch_contour.append("r")
	return np.array(pitch_contour)

def markov_chain(track):
	p = octave_abstraction(track)
	
	transitions = np.zeros((12, 12))
	for i in range(len(p) - 1):
		note_from = p[i]
		note_to = p[i+1]
		transitions[note_from, note_to] += 1
	
	for i in range(len(transitions)):
		total = np.sum(transitions[i])
		if total != 0:
			transitions[i] /= total
	
	return transitions

