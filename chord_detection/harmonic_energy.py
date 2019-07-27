import numpy
import random
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
from .multipitch import Multipitch, Chromagram
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

    def display_name(self):
        return "Harmonic Energy (Stark, Plumbley)"

    def compute_pitches(self):
        # first, signal frame
        self.x_frame = self.x[: self.frame_size]

        # then hamming
        self.x_frame = self.x_frame * scipy.signal.hamming(self.frame_size)

        # then sqrt of magnitude of DFT
        self.x_dft = numpy.sqrt(numpy.absolute(numpy.fft.rfft(self.x_frame)))

        # chroma vector calculation
        chromagram = Chromagram()

        # first C = C3 aka 130.81 Hz
        notes = list(gen_octave(130.81))

        divisor_ratio = (self.fs / 4.0) / self.frame_size
        self.dft_maxes = []

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
                        curr_ = self.x_dft[k]
                        if curr_ > x_dft_max:
                            x_dft_max = curr_
                            best_ind = k

                    note_sum += x_dft_max * (1.0 / harmonic)
                    self.dft_maxes.append((k0, best_ind, k1))
                chroma_sum += note_sum
            chromagram[n] += chroma_sum

        chromagram.normalize()
        return chromagram

    def display_plots(self):
        pltlen = self.frame_size
        samples = numpy.arange(pltlen)
        dftlen = int(self.x_dft.shape[0] / 2)
        dft_samples = numpy.arange(dftlen)

        fig1, (ax1, ax2) = plt.subplots(2, 1)

        ax1.set_title("x[n] - {0}".format(self.clip_name))
        ax1.set_xlabel("n (samples)")
        ax1.set_ylabel("amplitude")
        ax1.plot(samples, self.x[:pltlen], "b", alpha=0.3, linestyle=":", label="x[n]")
        ax1.plot(
            samples,
            self.x_frame[:pltlen],
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
        ax2.plot(dft_samples, self.x_dft[:dftlen], "b", alpha=0.5, label="X(n)")
        for i, dft_max in enumerate(self.dft_maxes):
            left, mid, right = dft_max
            ax2.plot(left, self.x_dft[:dftlen][left], "rx")
            ax2.plot(mid, self.x_dft[:dftlen][mid], "go")
            ax2.plot(right, self.x_dft[:dftlen][right], color="purple", marker="x")
            pitch = self.fs / mid
            note = freq_to_note(pitch)
            pitch = round(pitch, 2)

            if (i % 17) == 0:
                # displaying too many of these clutters the graph
                ax2.text(
                    mid, 1.2 * self.x_dft[:dftlen][mid], "{0}\n{1}".format(pitch, note)
                )

        ax2.grid()
        ax2.legend(loc="upper right")

        plt.show()
