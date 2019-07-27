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
        frame_size=2 * 8192,
        power=0.67,
        channels=70,
        epsilon0=2.3,
        epsilon1=0.39,
        peak_thresh=0.5,
        peak_min_dist=10,
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
        self.rand_channels = {}
        for i in range(5):
            chosen_idx = random.randrange(0, len(self.channels))
            self.rand_channels[chosen_idx] = {'fc': self.channels[chosen_idx], 'x_haf': None}

    def display_name(self):
        return "Iterative F0 (Klapuri, Anssi)"

    def compute_pitches(self):
        self.yc = []

        for i, fc in enumerate(self.channels):
            yc = _auditory_filterbank(self.x, self.fs, fc)

            yc = wfir(yc, self.fs, 12)  # dynamic level compression
            yc[yc < 0] = -yc[yc < 0]  # full-wave rectification

            yc = (
                yc + lowpass_filter(yc, self.fs, fc)
            ) / 2.0  # sum with low-pass filtered version of self at center-channel frequency

            yc *= scipy.signal.hamming(self.x.shape[0])
            if i in self.rand_channels:
                self.rand_channels[i]['x_haf'] = yc.copy()
            self.yc.append(yc)

        self.Ut = sacf(self.yc, k=self.power)

        chromagram = Chromagram()

        peak_indices = peakutils.indexes(
            self.Ut, thres=self.peak_thresh, min_dist=self.peak_min_dist
        )

        peak_indices_interp = peakutils.interpolate(
            numpy.arange(self.Ut.shape[0]), self.Ut, ind=peak_indices
        )
        for i, tau in enumerate(peak_indices_interp):
            pitch = self.fs / tau
            note = freq_to_note(pitch)
            chromagram[note] += self.Ut[peak_indices[i]]

        chromagram.normalize()
        return chromagram

    def display_plots(self):
        plot_len = min(self.Ut.shape[0], self.x.shape[0])
        samples = numpy.arange(plot_len)

        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1)
        i = 0

        ax1.set_title("x[n] - {0}".format(self.clip_name))
        ax1.set_xlabel("n (samples)")
        ax1.set_ylabel("amplitude")
        ax1.plot(samples, self.x[:plot_len], "C{0}".format(i), alpha=0.75, linestyle='--', label="x[n]")
        for x, dct in self.rand_channels.items():
            ax1.plot(samples, dct['x_haf'][:plot_len], "C{0}".format(i), alpha=0.75, linestyle=':', label="x[n] auditory filterbank, {0}Hz".format(dct['fc']))
            i += 1
        ax1.grid()
        ax1.legend(loc="upper right")

        ax2.set_title("yc by channel (randomly selected)")
        ax2.set_xlabel("n (samples)")
        ax2.set_ylabel("amplitude")

        for x in self.rand_channels:
            xaxis = int(self.x.shape[0])
            ax2.plot(
                numpy.arange(xaxis),
                self.yc[x][:xaxis],
                color="C{0}".format(i),
                linestyle="--",
                alpha=0.5,
                label="{0}Hz".format(round(self.channels[x], 2)),
            )
        i += 1

        ax2.grid()
        ax2.legend(loc="upper right")

        ax3.set_title("Ut")
        ax3.set_xlabel("n (samples)")
        ax3.set_ylabel("|amplitude|^p")
        ax3.plot(
            samples, self.Ut[:plot_len], "g", linestyle="--", alpha=0.5, label="Ut"
        )

        ax3.grid()
        ax3.legend(loc="upper right")

        plt.axis("tight")
        fig1.tight_layout()
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
