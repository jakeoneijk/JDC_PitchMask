class PitchGaussian():
    def file_path_to_narray(self,filepath):
        narray = np.loadtxt(filepath, delimiter=" ")
        print("debug")