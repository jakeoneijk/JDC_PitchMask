import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def stft(filePath,samplingRate,numFft,hopLength,winLength):
    x, fs = librosa.load(filePath,sr=samplingRate)
    return librosa.core.stft(x,n_fft=numFft, hop_length=hopLength, win_length=winLength,window='hann')

test_file_path = './Data/mirex05TrainFiles/train01.wav'
X = stft(test_file_path,44100,1024,944,1024)

test_pitch_path = './Data/labrosa_pitch/pitch_train01.wav.txt'
test_pitch = np.loadtxt('./Data/labrosa_pitch/pitch_train01.wav.txt', delimiter=" ")

S = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(15,5))
librosa.display.specshow(S, sr=fs, hop_length=441, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.show()
print("debug")