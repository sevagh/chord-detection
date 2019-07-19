import numpy
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
from .multipitch import Multipitch
from .notes import freq_to_note, NOTE_NAMES


class MultipitchHarmonicEnergy(Multipitch):
    def __init__(self, audio_path, ham_ms=46.4):
        super().__init__(audio_path)
        self.ham_samples = int(self.fs * ham_ms / 1000.0)

    def compute_pitches(self):
        x_dft = numpy.sqrt(numpy.fft.rfft(self.x))
        chromagram = {}
        return chromagram

    def display_plots(self):
        pass
