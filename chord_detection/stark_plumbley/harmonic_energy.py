import numpy
import random
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
from ..multipitch import Multipitch
from ..chromagram import Chromagram
from ..dsp.frame import frame_cutter
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

    @staticmethod
    def display_name():
        return "Harmonic Energy (Stark, Plumbley)"

    @staticmethod
    def method_number():
        return 2

    def compute_pitches(self, display_plot_frame=-1):
        # first C = C3
        notes = librosa.cqt_frequencies(12, fmin=librosa.note_to_hz('C3'))

        divisor_ratio = (self.fs / 4.0) / self.frame_size
        self.dft_maxes = []

        overall_chromagram = Chromagram()

        for frame, x in enumerate(frame_cutter(self.x, self.frame_size)):
            chromagram = Chromagram()
            x = x * scipy.signal.hamming(self.frame_size)
            x_dft = numpy.sqrt(numpy.absolute(numpy.fft.rfft(x)))
            for n in range(12):
                chroma_sum = 0.0
                for octave in range(1, self.num_octave + 1):
                    note_sum = 0.0
                    for harmonic in range(1, self.num_harmonic + 1):
                        x_dft_max = float("-inf")  # sentinel

                        k_prime = numpy.round(
                            (notes[n] * octave * harmonic) / divisor_ratio
                        )
                        k0 = int(k_prime - self.num_bins * harmonic)
                        k1 = int(k_prime + self.num_bins * harmonic)

                        best_ind = None
                        for k in range(k0, k1):
                            curr_ = x_dft[k]
                            if curr_ > x_dft_max:
                                x_dft_max = curr_
                                best_ind = k

                        note_sum += x_dft_max * (1.0 / harmonic)
                        self.dft_maxes.append((k0, best_ind, k1))
                    chroma_sum += note_sum
                chromagram[n] += chroma_sum

            overall_chromagram += chromagram

            if frame == display_plot_frame:
                _display_plots(self.clip_name, self.fs, self.frame_size, x_dft, self.x, x, self.dft_maxes)
        return overall_chromagram

def _display_plots(clip_name, fs, frame_size, x_dft, x, x_frame, dft_maxes):
    pltlen = frame_size
    samples = numpy.arange(pltlen)
    dftlen = int(x_dft.shape[0] / 2)
    dft_samples = numpy.arange(dftlen)

    fig1, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title("x[n] - {0}".format(clip_name))
    ax1.set_xlabel("n (samples)")
    ax1.set_ylabel("amplitude")
    ax1.plot(samples, x[:pltlen], "b", alpha=0.3, linestyle=":", label="x[n]")
    ax1.plot(
        samples,
        x_frame[:pltlen],
        "r",
        alpha=0.4,
        linestyle="--",
        label="x[n], frame + ham",
    )
    ax1.grid()
    ax1.legend(loc="upper right")

    ax2.set_title("X (DFT)")
    ax2.set_xlabel("fft bin")
    ax2.set_ylabel("magnitude")
    ax2.plot(dft_samples, x_dft[:dftlen], "b", alpha=0.5, label="X(n)")
    for i, dft_max in enumerate(dft_maxes):
        left, mid, right = dft_max
        ax2.plot(left, x_dft[:dftlen][left], "rx")
        ax2.plot(mid, x_dft[:dftlen][mid], "go")
        ax2.plot(right, x_dft[:dftlen][right], color="purple", marker="x")
        pitch = fs / mid
        note = librosa.hz_to_note(pitch, octave=False)
        pitch = round(pitch, 2)

        if (i % 17) == 0:
            # displaying too many of these clutters the graph
            ax2.text(
                mid, 1.2 * x_dft[:dftlen][mid], "{0}\n{1}".format(pitch, note)
            )

    ax2.grid()
    ax2.legend(loc="upper right")

    plt.show()
