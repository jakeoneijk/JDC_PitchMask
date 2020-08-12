import numpy as np

class PitchGaussian():
    def __init__(self):
        self.f0_array = None
        self.f0_array_transform = None
        self.max_frequency_index = 513

    def file_path_to_narray(self,filepath):
        narray = np.loadtxt(filepath, delimiter=" ")
        self.f0_array = narray[:,1]
        return self.f0_array
    
    def hz_to_bin(self,sr,fft_size):
        f0_bin = np.around((self.f0_array / sr) * fft_size)
        return f0_bin
    
    def matrix_fit_to_spectro(self,f0_bin):
        matrix_fit = np.zeros((513, 2480))
        for i in range(0,len(f0_bin)):
            time_index = int(i)
            frequency_index = int(f0_bin[i])
            while frequency_index < self.max_frequency_index and frequency_index != 0:
                matrix_fit[frequency_index,time_index] = 1
                frequency_index = frequency_index + int(f0_bin[i])
        self.f0_array_transform = matrix_fit

        return self.f0_array_transform