import os
import sys
import gzip
import joblib
import numpy as np

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

def feature_aggregator(track, extractor_name, aggregator_name):
	filename = handle_directories(track, extractor_name, aggregator_name)
	
	if aggregator_name == "bypass":
		return fe.feature_extractor(track, extractor_name).flatten()
	
	if not os.path.isfile(filename):
		print("computing", filename)
		track.local_feat = fe.feature_extractor(track, extractor_name)
		thismodule = sys.modules[__name__]
		global_feat = getattr(thismodule, aggregator_name)(track)
		track.unload()
		with gzip.GzipFile(filename, "wb", compresslevel=3) as fo:
			joblib.dump(global_feat, fo)
		return global_feat
	else:
		print("loading", filename)
		with gzip.GzipFile(filename, "rb") as fo:
			global_feat = joblib.load(fo)
			return global_feat

# =================================================================================================
# single gaussian methods
# =================================================================================================
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

