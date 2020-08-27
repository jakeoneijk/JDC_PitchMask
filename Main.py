import librosa
import librosa.display
import StftFitToModel
import PitchGaussian
import matplotlib.pyplot as plt
import numpy as np

class AppController():
    def __init__(self):
        self.stft_fit_to_model = StftFitToModel.StftFitToModel()
        self.pitch_gaussian = PitchGaussian.PitchGaussian(self.stft_fit_to_model.sampling_rate,self.stft_fit_to_model.number_fft,self.stft_fit_to_model.max_frequency_index)
        self.test_file_name = 'train02.wav'
        self.test_file_path = './Data/mirex05TrainFiles/' + self.test_file_name
        self.test_pitch_path = './Data/labrosa_pitch/pitch_'+self.test_file_name+'.txt'

    def stft_after_processing(self):
        X = self.stft_fit_to_model.down_sampling_and_stft(self.test_file_path)
        return self.stft_fit_to_model.after_processing(X)

    def pitch_gaussian_processing(self):
        self.pitch_gaussian.file_path_to_narray(self.test_pitch_path)
        return self.pitch_gaussian.matrix_fit_to_spectro(self.stft_fit_to_model.number_time_frame)

    def plot(self,X):
        S = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15,5))
        librosa.display.specshow(S, sr=self.stft_fit_to_model.sampling_rate, hop_length=self.stft_fit_to_model.hop_length, x_axis='time', y_axis='log')
        #plt.colorbar(format='%+2.0f dB')
        plt.show()
        print("debug")

    def main_control(self):
        stft_fit_to_model_spectro = self.stft_after_processing()
        making = self.pitch_gaussian_processing()
        masked_spectro = making * stft_fit_to_model_spectro
        self.stft_fit_to_model.inverse_stft(masked_spectro)
        self.plot(masked_spectro)

        print("debug")

if __name__ == "__main__":
    app_controller = AppController()
    app_controller.main_control()

