import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

class Track:
	def __init__(self, path, class_, basename, extension, index):
		# constants
		self.samplerate = 44100
		
		# attributes
		self.path = path
		self.class_ = class_
		self.basename = basename
		self.extension = extension
		self.index = index
		self.filename = os.path.join(self.path, "audio", self.class_, self.basename + self.extension)
		
		# holders
		self.audio = None
		self.local_feat = None
		self.global_feat = None
	
	def load(self):
		audio, sr = librosa.core.load(self.filename, sr=self.samplerate)
		assert sr == self.samplerate
		
		audio = librosa.to_mono(audio)
		assert audio.ndim == 1
		
		# data cleaning
		audio, index = librosa.effects.trim(audio) # trim on both ends
		audio = audio - np.mean(audio) # filter out the DC component
		
		self.audio = audio
	
	def unload(self):
		self.audio = None
		self.local_feat = None
		self.global_feat = None
	
	def __str__(self):
		return str(self.index) + ": " + self.class_ + "/" + self.basename + " (~" + str(int(len(self.audio) / self.samplerate)) + "sec)"
	
	def __hash__(self):
		return hash(self.class_ + "/" + self.basename)
	
	def __eq__(self, other):
		return self.class_ == other.class_ and self.basename == other.basename

