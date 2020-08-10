import librosa
import librosa.display
import matplotlib.pyplot as plt

test_file_path = './Data/mirex05TrainFiles/train01.wav'
x, fs = librosa.load(test_file_path,  sr=44100)
X = librosa.core.stft(x,n_fft=1024, hop_length=512, win_length=1024)
x_shape = X.shape
S = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(15,5))
librosa.display.specshow(S, sr=fs, hop_length=441, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.show()
print("debug")