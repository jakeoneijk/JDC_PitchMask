import numpy as np
import math


class PitchGaussian():
    def __init__(self,sampling_rate,fft_size,max_frequency_index):
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size
        self.max_frequency_index = max_frequency_index
        self.f0_array = None
        self.kernel_size = 100
        self.sigma = 10
        self.W = 10

    def file_path_to_narray(self,filepath):
        narray = np.loadtxt(filepath, delimiter=" ")
        self.f0_array = narray[:,1]
        return self.f0_array

    def hz_to_bin(self,hz):
        return int((hz / self.sampling_rate) * self.fft_size)

    def bin_to_hz(self,bin):
        return (bin / self.fft_size) * self.sampling_rate

    def gaussian_function_array(self,mean,variance,x):
        denominator = math.sqrt(float(2 * math.pi * (variance ** 2)))
        return self.W * np.exp(-1 * (((x - mean) ** 2)/(2 * (variance ** 2)))) / denominator

    def hz_to_gaussian_kernel(self,hz):
        bin_of_hz = self.hz_to_bin(hz)
        start_index = max((bin_of_hz - int(self.kernel_size / 2)),0)
        end_index = min((bin_of_hz + int(self.kernel_size / 2)),self.max_frequency_index)

        bin_array = np.arange(start_index , end_index)
        bin_to_hz_array = self.bin_to_hz(bin_array)
        gaussian_value = self.gaussian_function_array(hz,self.sigma,bin_to_hz_array)
        return gaussian_value,start_index,end_index

    def matrix_fit_to_spectro(self,number_time_frame):
        f0_array_transform = np.zeros((self.max_frequency_index, number_time_frame))
        for i in range(0,len(self.f0_array)):
            print(i)
            time_index = int(i)
            masking_hz = self.f0_array[i]
            while masking_hz < (self.sampling_rate/2) and masking_hz != 0:
                gaussian_array,start_index,end_index = self.hz_to_gaussian_kernel(masking_hz)
                f0_array_transform[start_index:end_index,time_index] = f0_array_transform[start_index:end_index,time_index] + gaussian_array
                masking_hz = masking_hz + self.f0_array[i]

        return f0_array_transform