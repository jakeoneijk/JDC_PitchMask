import numpy as np
import librosa
import scipy
from scipy import io
from scipy.io.wavfile import write

class SpecExtraction():
    def __init__(self):
        self.sampling_rate = 44100
        self.down_sampling_rate = 8000
        self.number_fft = 1024
        self.window_length = 1024
        self.hop_length = 80

    def down_sampling(self,file_name):
        y, sr = librosa.load(file_name, sr=self.sampling_rate)
        return librosa.resample(y, sr, self.down_sampling_rate)

    def down_sampling_and_stft(self,file_name):
        down_sample = self.down_sampling(file_name)
        X = librosa.core.stft(down_sample, n_fft=self.number_fft, hop_length=self.hop_length, win_length=self.window_length)
        return X

    def after_processing(self,X,win_size):
        x_spec = np.abs(X)
        x_spec  = librosa.core.power_to_db(x_spec,ref=np.max)
        x_spec = x_spec.astype(np.float32)
        num_frames = x_spec.shape[1]
        padNum = num_frames % win_size
        if padNum != 0:
            len_pad = win_size - padNum
            padding_feature = np.zeros(shape=(513, len_pad))
            x_spec = np.concatenate((x_spec, padding_feature), axis=1)
            num_frames = num_frames + len_pad
        return x_spec
    
    def inverse_stft(self,stft_mat):
        iStftMat = librosa.core.istft(stft_mat, hop_length=self.hop_length)
        write("./testOut.wav", self.down_sampling_rate, iStftMat)


