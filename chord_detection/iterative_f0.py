import numpy
import math
import scipy
import scipy.signal
import scipy.fftpack
import librosa
from numba import jit, njit
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
        power=1,
        channels=70,
        epsilon0=2.3,
        epsilon1=0.39,
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

    def compute_pitches(self):
        self.ytc = [[None] * len(self.channels)] * self.num_frames

        for i, fc in enumerate(self.channels):
            yc = _auditory_filterbank(self.x, self.fs, fc)

            yc = wfir(yc, self.fs, 12)  # dynamic level compression
            yc[yc < 0] = -yc[yc < 0]  # full-wave rectification

            yc = (
                yc + lowpass_filter(yc, self.fs, fc)
            ) / 2.0  # sum with low-pass filtered version of self at center-channel frequency

            for j, yc_ in enumerate(
                numpy.split(yc, self.num_frames)
            ):  # split yc into frames
                if len(yc_) < self.frame_size:
                    yc_ = numpy.concatenate(
                        (yc_, numpy.zeros(self.frame_size - len(yc_)))
                    )

                yc_ *= scipy.signal.hamming(self.frame_size)
                yc_ = numpy.concatenate((yc_, numpy.zeros(self.frame_size)))

                self.ytc[j][i] = yc_

        self.Ut = []
        for frame, yc in enumerate(self.ytc):
            Ut_ = sacf(yc, k=self.power)
            self.Ut.append(Ut_)

        chromagram = Chromagram()

        for frame, U in enumerate(self.Ut):
            peaks, _ = scipy.signal.find_peaks(U)
            print(peaks)
            peak_prominence, _, _ = scipy.signal.peak_prominences(U, peaks)
            print(peak_prominence)

            max_peak_prominences = numpy.argpartition(peak_prominence, -1)[-1:]
            print(max_peak_prominences)

            for i in max_peak_prominences:
                print(i)
                print(peaks[i])
                pitch = self.fs / peaks[i]
                note = freq_to_note(pitch)
                chromagram[note] += peak_prominence[i]

        chromagram.normalize()
        return chromagram

    def display_plots(self):
        # for frame, yc in enumerate(iterativef0.ytc):
        #    Ut = self.Ut[frame]
        plot_len = min(self.Ut[0].shape[0], self.x.shape[0])
        samples = numpy.arange(plot_len)

        fig1, (ax1, ax2) = plt.subplots(2, 1)

        ax1.set_title("x[n] - {0}".format(self.clip_name))
        ax1.set_xlabel("n (samples)")
        ax1.set_ylabel("amplitude")
        ax1.plot(samples, self.x[:plot_len], "b", alpha=0.8, label="x[n]")
        ax1.grid()
        ax1.legend(loc="upper right")

        ax2.set_title("Ut")
        ax2.set_xlabel("n (samples)")
        ax2.set_ylabel("amplitude")
        ax2.plot(
            samples, self.Ut[0][:plot_len], "g", linestyle="--", alpha=0.5, label="sacf"
        )

        ax2.grid()
        ax2.legend(loc="upper right")

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