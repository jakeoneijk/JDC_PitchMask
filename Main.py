import librosa
import librosa.display
import StftFitToModel
import PitchGaussian
import matplotlib.pyplot as plt

class AppController():
    def __init__(self):
        self.stft_fit_to_model = StftFitToModel.StftFitToModel()
        self.pitch_gaussian = PitchGaussian.PitchGaussian(self.stft_fit_to_model.sampling_rate,self.stft_fit_to_model.number_fft,self.stft_fit_to_model.max_frequency_index)
        self.test_file_name = 'sun'
        self.extension = '.mp3'
        self.test_file_path = './Data/' + self.test_file_name + self.extension
        self.test_pitch_path = './Data/'+self.test_file_name + self.extension+'_pitch.txt'
        self.test_is_vocal_path = './Data/'+self.test_file_name + self.extension+'_is_vocal.txt'

    def stft_after_processing(self):
        self.stft_fit_to_model.test_property()
        X , pitch_data_length = self.stft_fit_to_model.stft_and_get_num_time_frame(self.test_file_path)
        return X , pitch_data_length , X.shape[1]

    def pitch_gaussian_processing(self,pitch_path,is_vocal_path,pitch_data_length, num_spec,isRef):
        self.pitch_gaussian.file_path_to_narray(pitch_path,is_vocal_path,isRef)
        return self.pitch_gaussian.matrix_fit_to_spectro(pitch_data_length, num_spec)

    def plot(self,X):
        S = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15,5))
        librosa.display.specshow(S, sr=self.stft_fit_to_model.sampling_rate, hop_length=self.stft_fit_to_model.hop_length, x_axis='time', y_axis='log')

    def main_control(self):
        stft_fit_to_model_spectro , pitch_data_length , num_spec = self.stft_after_processing()
        making = self.pitch_gaussian_processing(self.test_pitch_path,self.test_is_vocal_path, pitch_data_length, num_spec,False)
        masked_spectro = making * stft_fit_to_model_spectro

        self.stft_fit_to_model.inverse_stft(stft_fit_to_model_spectro, "(original)")
        self.stft_fit_to_model.inverse_stft_griffin_lim(masked_spectro,"(masked griffin)")
        self.stft_fit_to_model.inverse_stft(masked_spectro, "(masked)")
        self.plot(masked_spectro)
        self.plot(making)
        print("debug")

if __name__ == "__main__":
    app_controller = AppController()
    app_controller.main_control()


