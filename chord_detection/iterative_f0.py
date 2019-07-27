import numpy
import math
import random
import scipy
import scipy.signal
import scipy.fftpack
import peakutils
import librosa
import matplotlib.pyplot as plt
from .multipitch import Multipitch, Chromagram
from .notes import freq_to_note, gen_octave, NOTE_NAMES
from .wfir import wfir
from .esacf import lowpass_filter, sacf
from collections import OrderedDict


class MultipitchIterativeF0(Multipitch):
    def __init__(
        self,
        audio_path,
        frame_size=8192,
        power=0.67,
        channels=70,
        epsilon0=2.3,
        epsilon1=0.39,
        peak_thresh=0.5,
        peak_min_dist=10,
        harmonic_multiples_elim=10,
    ):
        super().__init__(audio_path)
        self.frame_size = frame_size
        self.power = power
        num_frames = float(len(self.x)) / float(frame_size)
        num_frames = int(math.ceil(num_frames))
        self.num_frames = num_frames
        pad = int(num_frames * frame_size - len(self.x))
        self.x = numpy.concatenate((self.x, numpy.zeros(pad)))
        self.channels = [
            229 * (10 ** ((epsilon1 * c + epsilon0) / 21.4) - 1)
            for c in range(channels)
        ]
        self.peak_thresh = peak_thresh
        self.peak_min_dist = peak_min_dist
        self.harmonic_multiples_elim = harmonic_multiples_elim

    def display_name(self):
        return "[INCOMPLETE] Iterative F0 (Klapuri, Anssi)"

    def compute_pitches(self):
        self.yc = []

        for i, fc in enumerate(self.channels):
            yc = self.x[: self.frame_size]

            yc = _auditory_filterbank(yc, self.fs, fc)

            yc = wfir(yc, self.fs, 12)  # dynamic level compression
            yc[yc < 0] = -yc[yc < 0]  # full-wave rectification

            yc = (
                yc + lowpass_filter(yc, self.fs, fc)
            ) / 2.0  # sum with low-pass filtered version of self at center-channel frequency

            yc = numpy.append(
                yc * scipy.signal.hamming(self.frame_size), numpy.zeros(self.frame_size)
            )

            self.yc.append(yc)

        # while method 1/ESACF uses time stretching to eliminate harmonics, here we'll be iteratively eliminating the dominant f0?
        self.Ut = sacf(self.yc, k=self.power)

        chromagram = Chromagram()

        while True:
            peak_indices = peakutils.indexes(
                self.Ut, thres=self.peak_thresh, min_dist=self.peak_min_dist
            )

            chosen_tau = None
            if len(peak_indices) > 0:
                peak_indices_interp = peakutils.interpolate(
                    numpy.arange(self.Ut.shape[0]), self.Ut, ind=peak_indices
                )
                chosen_tau = peak_indices_interp[0]
                real_tau = peak_indices[0]

                pitch = self.fs / chosen_tau
                note = freq_to_note(pitch)
                chromagram[note] += self.Ut[real_tau]
                for i in range(1, self.harmonic_multiples_elim):
                    index = i * real_tau
                    if index > self.Ut.shape[0]:
                        break

                    # eliminate harmonic multiples of the same f0
                    self.Ut[index] -= self.Ut[index]
            else:
                break

        chromagram.normalize()
        return chromagram

    def display_plots(self):
        fig1, (ax1, ax2) = plt.subplots(2, 1)

        ax1.set_title("x[n] - {0}".format(self.clip_name))
        ax1.set_xlabel("n (samples)")
        ax1.set_ylabel("amplitude")
        ax1.plot(
            numpy.arange(self.frame_size),
            self.x[: self.frame_size],
            "b",
            alpha=0.75,
            linestyle="--",
            label="x[n]",
        )

        ax1.grid()
        ax1.legend(loc="upper right")

        ax2.set_title(r"$y_c$[n], auditory filterbank")
        ax2.set_xlabel("n (samples)")
        ax2.set_ylabel("amplitude")

        for i, x in enumerate(
            [random.randrange(0, len(self.channels)) for _ in range(10)]
        ):
            ax2.plot(
                numpy.arange(self.frame_size),
                self.yc[x][: self.frame_size],
                color="C{0}".format(i),
                linestyle="--",
                alpha=0.5,
                label="{0}Hz".format(round(self.channels[x], 2)),
            )
        i += 1

        ax2.grid()
        ax2.legend(loc="upper right")

        plt.show()


def _auditory_filterbank(x, fc, fs):
    J = 4

    # bc3db = -3/J dB
    A = numpy.exp(-(3 / J) * numpy.pi / (fs * numpy.sqrt(2 ** (1 / J) - 1)))

    cos_theta1 = (1 + A * A) / (2 * A) * numpy.cos(2 * numpy.pi * fc / fs)
    cos_theta2 = (2 * A) / (1 + A * A) * numpy.cos(2 * numpy.pi * fc / fs)
    rho1 = (1 / 2) * (1 - A * A)
    rho2 = (1 - A * A) * numpy.sqrt(1 - cos_theta2 ** 2)

    resonator_1_b = [rho1, 0, -rho1]
    resonator_1_a = [1, -A * cos_theta1, A * A]

    resonator_2_b = [rho2]
    resonator_2_a = [1, -A * cos_theta2, A * A]

    x = scipy.signal.lfilter(resonator_1_b, resonator_1_a, x)
    x = scipy.signal.lfilter(resonator_1_b, resonator_1_a, x)
    x = scipy.signal.lfilter(resonator_2_b, resonator_2_a, x)
    x = scipy.signal.lfilter(resonator_2_b, resonator_2_a, x)

    return x
