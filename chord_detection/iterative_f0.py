import numpy
import math
import random
import typing
import scipy
import scipy.signal
import scipy.fftpack
import peakutils
import librosa
import matplotlib.pyplot as plt
from numba import njit, jit
from .multipitch import Multipitch
from .chromagram import Chromagram
from .notes import freq_to_note, gen_octave, NOTE_NAMES
from .wfir import wfir
from .esacf import lowpass_filter
from collections import OrderedDict


class MultipitchIterativeF0(Multipitch):
    def __init__(
        self,
        audio_path,
        frame_size=8192,
        power=1.0,
        channels=70,
        zeta0=2.3,
        zeta1=0.39,
        epsilon1=20,
        epsilon2=320,
        peak_thresh=0.5,
        peak_min_dist=10,
        harmonic_multiples_elim=5,
        M=20,
        delta_tau=20,
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
            229 * (10 ** ((zeta1 * c + zeta0) / 21.4) - 1) for c in range(channels)
        ]
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.peak_thresh = peak_thresh
        self.peak_min_dist = peak_min_dist
        self.harmonic_multiples_elim = harmonic_multiples_elim
        self.M = M
        self.delta_tau = delta_tau

    def display_name(self):
        return "Iterative F0 (Klapuri, Anssi)"

    def compute_pitches(self):
        self.ytc = [
            [None for _ in range(len(self.channels))] for _ in range(self.num_frames)
        ]

        for i, fc in enumerate(self.channels):
            for j, x_t in enumerate(numpy.split(self.x, self.num_frames)):
                yc = _auditory_filterbank(x_t, self.fs, fc)
                yc = wfir(yc, self.fs, 12)  # dynamic level compression
                yc[yc < 0] = -yc[yc < 0]  # full-wave rectification
                yc = (
                    yc + lowpass_filter(yc, self.fs, fc)
                ) / 2.0  # sum with low-pass filtered version of self at center-channel frequency

                yc = numpy.append(
                    yc * scipy.signal.hamming(self.frame_size),
                    numpy.zeros(self.frame_size),
                )

                self.ytc[j][i] = yc

        self.Ut = [_bandwise_summary_spectrum(yc, k=self.power) for yc in self.ytc]

        chromagram = Chromagram()

        K = self.Ut[0].shape[0]
        num_candidates = int(K / 2)

        self.salience_t_tau = [
            numpy.zeros(num_candidates) for _ in range(self.num_frames)
        ]

        for frame, Ut in enumerate(self.Ut):
            for tau in range(
                1, num_candidates
            ):  # maximum tau candidate is capped by nyquist
                for m in range(1, self.M):
                    self.salience_t_tau[frame][tau] += _weight(
                        tau, self.fs, self.epsilon1, self.epsilon2, m
                    ) * _max_Ut(Ut, tau, self.delta_tau, m, K)

        for frame, salience_t in enumerate(self.salience_t_tau):
            max_tau = numpy.argmax(salience_t)
            note = freq_to_note(self.fs / max_tau)
            chromagram[note] += salience_t[max_tau]

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
                self.ytc[0][x][: self.frame_size],
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


"""
i would've used the sacf() function here from method 1
but the IFFT is specific to that method and we're using weighted salience for periodicity analysis here
"""


def _bandwise_summary_spectrum(
    x_channels: typing.List[numpy.ndarray], k=None
) -> numpy.ndarray:
    # k is same as p (power) in the Klapuri/Ansi paper
    if not k:
        k = 0.67

    running_sum = numpy.zeros(x_channels[0].shape[0])

    for xc in x_channels:
        running_sum += numpy.abs(numpy.fft.fft(xc)) ** k

    return running_sum


def _weight(tau, fs, epsilon1, epsilon2, m):
    return (fs / tau + epsilon1) / (m * fs / tau + epsilon2)


@njit
def _max_Ut(Ut, tau, delta_tau, m, K):
    Ut_max = 0.0
    for tau_adjusted in numpy.linspace(tau + delta_tau / 2, tau - delta_tau / 2):
        if tau_adjusted == 0.0:
            continue
        k_tau_m = int(round(m * K / tau_adjusted))
        if 0 <= k_tau_m < Ut.shape[0]:
            Ut_max = max(Ut[k_tau_m], Ut_max)
    return Ut_max
