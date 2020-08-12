import numpy as np
import cv2

class PitchGaussian():
    def __init__(self):
        self.f0_array = None
        self.f0_array_transform = np.zeros((513, 2480))
        self.max_frequency_index = 513
        self.kernel_size = 200
        self.sigma = 3

    def file_path_to_narray(self,filepath):
        narray = np.loadtxt(filepath, delimiter=" ")
        self.f0_array = narray[:,1]
        return self.f0_array
    
    def hz_to_bin(self,sr,fft_size):
        f0_bin = np.around((self.f0_array / sr) * fft_size)
        return f0_bin

    def matrix_fit_to_spectro(self,f0_bin):
        kernel = cv2.getGaussianKernel(self.kernel_size, self.sigma)

        for i in range(0,len(f0_bin)):
            print(i)
            time_index = int(i)
            frequency_index = int(f0_bin[i])
            while frequency_index < self.max_frequency_index and frequency_index != 0:
                for k in range(0,self.kernel_size):
                    fre_index = frequency_index - int(self.kernel_size/2) + k
                    if fre_index >= 0 and fre_index < 513:
                        self.f0_array_transform[fre_index,time_index] = self.f0_array_transform[fre_index,time_index]+ kernel[k][0]
                frequency_index = frequency_index + int(f0_bin[i])

        return self.f0_array_transform