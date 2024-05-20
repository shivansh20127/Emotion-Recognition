import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import opensmile
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa

# Set the current working directory
os.chdir("D:/Desktop/IIIT D Course Material/5th sem/ML/Project")

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

emotion_dict = { 0: "neutral", 1:"calm", 2:"happy", 3:"sad", 4:"angry", 5:"fearful", 6:"disgust", 7:"surprised"}

# Set the sample rate and duration
sample_rate = 16000
duration = 5  # in seconds

print("Recording...")

# Record audio
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()

print("Recording complete")

#Trim silence
recording, _ = librosa.effects.trim(recording, top_db=20, frame_length=512, hop_length=64)

# Save the recording as a .wav file
file_path = "demo.wav"
wav.write(file_path, sample_rate, recording)

# Load scaler.pkl and model.pkl files
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

print("Model and scaler loaded, predicting...")

features = smile.process_file(file_path).values.reshape((1, -1))
features_scaled = scaler.transform(features)
pred = model.predict(features_scaled)

print(emotion_dict[pred.sum()])
