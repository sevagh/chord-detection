import numpy
import scipy
import scipy.signal
import librosa
from numba import jit
import matplotlib.pyplot as plt
from .multipitch import Multipitch
from .notes import freq_to_note, gen_octave, NOTE_NAMES
from collections import OrderedDict


class MultipitchHarmonicEnergy(Multipitch):
    def __init__(
        self, audio_path, frame_size=8192, num_harmonic=2, num_octave=2, num_bins=2
    ):
        super().__init__(audio_path)
        self.frame_size = frame_size
        self.num_harmonic = num_harmonic
        self.num_octave = num_octave
        self.num_bins = num_bins

    @jit(nopython=False)
    def compute_pitches(self):
        # first, signal frame
        x = self.x[: self.frame_size]

        # then hamming
        x = x * scipy.signal.hamming(self.frame_size)

        # then sqrt of magnitude of DFT
        x_dft = numpy.sqrt(numpy.absolute(numpy.fft.rfft(x)))

        # chroma vector calculation
        Cn = [0.0] * 12

        # my own interpretation - the chromagram
        chromagram = OrderedDict()
        for n in NOTE_NAMES:
            chromagram[n] = 0.0

        # first C = C3 aka 130.81 Hz
        notes = list(gen_octave(130.81))

        divisor_ratio = (self.fs / 4.0) / self.frame_size

        for n in range(12):
            chroma_sum = 0.0
            for octave in range(1, self.num_octave + 1):
                note_sum = 0.0
                for harmonic in range(1, self.num_harmonic + 1):
                    x_dft_max = float('-inf') # sentinel

                    k_prime = numpy.round(
                        (notes[n] * octave * harmonic) / divisor_ratio
                    )
                    k0 = int(k_prime - self.num_bins * harmonic)
                    k1 = int(k_prime + self.num_bins * harmonic)

                    for k in range(k0, k1):
                        x_dft_max = max(x_dft[k], x_dft_max)

                    note_sum += x_dft_max * (1.0 / harmonic)
                chroma_sum += note_sum
            Cn[n] += chroma_sum
            chromagram[NOTE_NAMES[n]] = Cn[n]

        # normalize the chromagram
        chromagram_max = max(chromagram.values())
        for k in chromagram.keys():
            chromagram[k] = chromagram[k] / chromagram_max

        return chromagram

    def display_plots(self):
        pass
