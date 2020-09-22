import numpy as np
import librosa
import soundfile as sf
import time
from math import ceil
from matplotlib import pyplot as plt

class StftFitToModel():
    def __init__(self):
        self.sampling_rate = 44100
        self.number_fft = 2048
        self.max_frequency_index = int((self.number_fft / 2) + 1)
        self.window_length = 1764 #1764 #2048
        self.hop_length = 441 #441
        self.hop_length_for_pitch_data_length = int(self.sampling_rate / 100)
        self.model_input_size = 31
        self.window_type = 'hann'
        
    def test_property(self):
        window = librosa.filters.get_window(self.window_type,self.window_length)
        sum_for_test = np.zeros(10*self.window_length)
        for i in range(0,9*self.window_length,self.hop_length):
            sum_for_test[i:i+self.window_length] += window
        plt.plot(np.arange(len(sum_for_test)),sum_for_test)
        plt.show()
        print("debug")

    def stft_and_get_num_time_frame(self,file_name):
        y, sr = librosa.load(file_name, sr=self.sampling_rate)
        X = librosa.core.stft(y, n_fft=self.number_fft, hop_length=self.hop_length, win_length=self.window_length , window=self.window_type)
        pitch_data_length = int(ceil((y.size) / self.hop_length_for_pitch_data_length))
        return X , pitch_data_length
    
    def inverse_stft(self,stft_mat,name):
        istft_mat = librosa.core.istft(stft_mat, hop_length=self.hop_length, window = self.window_type , win_length=self.window_length)
        filename = "./Output/testOut"+ time.strftime('%c', time.localtime(time.time()))+"_"+name+".wav"
        sf.write(filename, istft_mat, self.sampling_rate)
        return istft_mat

    def inverse_stft_griffin_lim(self,stft_mat,name):
        istft_mat = librosa.griffinlim(abs(stft_mat), n_iter=50, hop_length=self.hop_length, win_length=self.window_length, window=self.window_type)
        filename = "./Output/testOut" + time.strftime('%c', time.localtime(time.time())) + "_" + name + ".wav"
        sf.write(filename, istft_mat, self.sampling_rate)
        return istft_mat

'''  
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
'''

