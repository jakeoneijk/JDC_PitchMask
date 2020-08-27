import numpy as np
import librosa
import scipy
from scipy import io
import soundfile as sf
import time

class StftFitToModel():
    def __init__(self):
        self.sampling_rate = 44100
        self.number_fft = 2048
        self.max_frequency_index = int((self.number_fft / 2) + 1)
        self.window_length = 2048
        self.hop_length = 441 #80
        self.number_time_frame = None
        self.model_input_size = 31
        

    def down_sampling(self,file_name):
        y, sr = librosa.load(file_name, sr=self.sampling_rate)
        return y

    def down_sampling_and_stft(self,file_name):
        down_sample = self.down_sampling(file_name)
        X = librosa.core.stft(down_sample, n_fft=self.number_fft, hop_length=self.hop_length, win_length=self.window_length)
        return X

    def after_processing(self,spectro):
        x_spec = spectro
        num_time_frames = x_spec.shape[1]

        padd_num = self.model_input_size - (num_time_frames % self.model_input_size)

        if padd_num != self.model_input_size:
            padding_feature = np.zeros(shape=(self.max_frequency_index, padd_num))
            x_spec = np.concatenate((x_spec, padding_feature), axis=1)
            num_time_frames = num_time_frames + padd_num

        self.number_time_frame = num_time_frames

        return x_spec
    
    def inverse_stft(self,stft_mat):
        iStftMat = librosa.core.istft(stft_mat, hop_length=self.hop_length)
        filename = "./testOut"+ time.strftime('%c', time.localtime(time.time()))+".wav"
        sf.write(filename, iStftMat, self.sampling_rate)
        return iStftMat


