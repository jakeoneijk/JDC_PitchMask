import librosa
import librosa.display
import SpecExtraction
import PitchGaussian
import matplotlib.pyplot as plt
import numpy as np

class AppController():
    def __init__(self):
        self.spec_extraction = SpecExtraction.SpecExtraction()
        self.pitch_gaussian = PitchGaussian.PitchGaussian()
        self.input_size = 31
        self.test_file_path = './Data/mirex05TrainFiles/train01.wav'
        self.test_pitch_path = './Data/labrosa_pitch/pitch_train01.wav.txt'
        self.X = None
        self.mask_f0 = None

    def stft_after_processing(self):
        X = self.spec_extraction.down_sampling_and_stft(self.test_file_path)
        self.X = self.spec_extraction.after_processing(X,31)

    def pitch_gaussian_processing(self):
        self.pitch_gaussian.file_path_to_narray(self.test_pitch_path)
        f0 = self.pitch_gaussian.hz_to_bin(self.spec_extraction.down_sampling_rate , self.spec_extraction.number_fft)
        self.mask_f0 = self.pitch_gaussian.matrix_fit_to_spectro(f0)

    def plot(self,X):
        S = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15,5))
        librosa.display.specshow(S, sr=8000, hop_length=80, x_axis='time', y_axis='log')
        #plt.colorbar(format='%+2.0f dB')
        plt.show()
        print("debug")

def main():
    app_controller = AppController()

    app_controller.stft_after_processing()

    app_controller.pitch_gaussian_processing()

    app_controller.spec_extraction.inverse_stft(app_controller.X * app_controller.mask_f0)

    app_controller.plot(app_controller.X * app_controller.mask_f0)
    print("debug")

if __name__ == "__main__":
    main()


