import os
import sys
import librosa
import numpy as np
from essentia.standard import FrameGenerator, Windowing, Spectrum, MFCC, PitchMelodia, PitchContourSegmentation

def handle_directories(track, extractor_name):
	base_dir = os.path.join(track.path, "local_features")
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
	
	return os.path.join(track_dir, extractor_name + ".npz")

# local features must be (m, n):
# m = number of frames
# n = dimension of the feature
def feature_extractor(track, extractor_name):
	filename = handle_directories(track, extractor_name)
	
	if not os.path.isfile(filename):
		print("computing", filename)
		track.load()
		thismodule = sys.modules[__name__]
		feature = getattr(thismodule, extractor_name)(track)
		track.unload()
		np.savez(filename, feature)
		return feature
	else:
		print("loading", filename)
		feature_fp =  np.load(filename)
		feature = feature_fp["arr_0"]
		feature_fp.close()
		return feature

# ************************************************************************************************
# features from librosa                                         https://github.com/librosa/librosa
# ************************************************************************************************
def melspectrogram(track):
	return librosa.feature.melspectrogram(track.audio, sr=track.samplerate).T

def chroma_stft(track):
	return librosa.feature.chroma_stft(track.audio, sr=track.samplerate).T

def chroma_cqt(track):
	return librosa.feature.chroma_cqt(track.audio, sr=track.samplerate).T

def chroma_cens(track):
	return librosa.feature.chroma_cens(track.audio, sr=track.samplerate).T

def tonnetz(track):
	return librosa.feature.tonnetz(track.audio, sr=track.samplerate).T

def spectral_centroid(track):
	return librosa.feature.spectral_centroid(track.audio, sr=track.samplerate).T

def spectral_bandwidth(track):
	return librosa.feature.spectral_bandwidth(track.audio, sr=track.samplerate).T

def spectral_contrast(track):
	return librosa.feature.spectral_contrast(track.audio, sr=track.samplerate).T

def spectral_flatness(track):
	return librosa.feature.spectral_flatness(track.audio).T

def spectral_rolloff(track):
	return librosa.feature.spectral_rolloff(track.audio, sr=track.samplerate).T

def rms(track):
	return librosa.feature.rms(track.audio).T

def zero_crossing_rate(track):
	return librosa.feature.zero_crossing_rate(track.audio).T

# ************************************************************************************************
# features from essentia                                           https://github.com/MTG/essentia
# ************************************************************************************************
def mfcc(track):
	frames = FrameGenerator(audio=track.audio)
	window = Windowing(type="hann")
	spectrum = Spectrum()
	
	spectrogram = []
	for frame in frames:
		frame_windowed = window(frame)
		frame_spectrum = spectrum(frame_windowed)
		spectrogram.append(frame_spectrum)
	
	spectrogram = np.array(spectrogram)
	mfcc = MFCC(sampleRate=track.samplerate, inputSize=513, numberCoefficients=20)
	mfccs = []
	for spectrum in spectrogram:
		mfcc_bands, mfcc_coeffs = mfcc(spectrum)
		mfccs.append(mfcc_coeffs[1:]) # drop the first coefficient
	return np.array(mfccs)

def pitch_contour_segmentation(track):
	pm = PitchMelodia(sampleRate=track.samplerate)
	pcs = PitchContourSegmentation(sampleRate=track.samplerate)
	pitch, pitchConfidence = pm(track.audio)
	onset, duration, pitches = pcs(pitch, track.audio)
	return pitches.reshape(len(pitches), 1)

