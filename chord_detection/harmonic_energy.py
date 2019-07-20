import numpy
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
from .multipitch import Multipitch
from .notes import freq_to_note, NOTE_NAMES
from collections import OrderedDict


class MultipitchHarmonicEnergy(Multipitch):
    def __init__(
        self,
        audio_path,
        ham_ms=46.4,
        frame_size=8192,
        num_harmonic=2,
        num_octave=2,
        num_bins=2,
    ):
        super().__init__(audio_path)
        self.ham_samples = int(self.fs * ham_ms / 1000.0)
        self.num_harmonic = num_harmonic
        self.num_octave = num_octave
        self.num_bins = num_bins
        self.frame_size = frame_size

    def compute_pitches(self):
        # first hamming
        diff = len(self.x) - self.ham_samples
        self.x = self.x * numpy.concatenate(
            (scipy.signal.hamming(self.ham_samples), numpy.zeros(diff))
        )

        # then sqrt of magnitude of DFT
        x_dft = numpy.sqrt(numpy.absolute(numpy.fft.rfft(self.x)))
        x_dft = x_dft[: self.frame_size]

        # chroma vector calculation
        Cn = [0.0] * 12

        # my own interpretation - the chromagram
        chromagram = OrderedDict()
        for n in NOTE_NAMES:
            chromagram[n] = 0.0

        for n in range(12):
            x_dft_max = float("-inf")  # sentinel value
            for octave in range(1, self.num_octave + 1):
                for harmonic in range(1, self.num_harmonic + 1):
                    freq = _lower_octave_frequencies(n)
                    k_prime = numpy.round(
                        (freq * octave * harmonic) / (self.fs / self.frame_size)
                    )
                    k0 = k_prime - self.num_bins * harmonic
                    k1 = k_prime + self.num_bins * harmonic

                    k = k0
                    while k <= k1:
                        k_idx = int(k)
                        if k_idx < 0:
                            k_idx = 0
                        if k_idx > len(x_dft) - 1:
                            x_idx = len(x_dft) - 1

                        if x_dft[k_idx] > x_dft_max:
                            x_dft_max = x_dft[k_idx]
                        x_dft_max = max(x_dft[k_idx], x_dft_max)
                        k += 1

                    Cn[n] += x_dft_max * (1 / harmonic)
                    chromagram[NOTE_NAMES[n]] += Cn[n]

        # normalize the chromagram
        chromagram_max = max(chromagram.values())
        for k in chromagram.keys():
            chromagram[k] = chromagram[k] / chromagram_max

        return chromagram

    def display_plots(self):
        pass


def _lower_octave_frequencies(n):
    fc3 = 130.81  # Hz
    return fc3 * (2.0 ** (n / 12))
