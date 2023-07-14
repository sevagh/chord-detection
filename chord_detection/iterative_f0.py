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
from chord_detection.multipitch import Multipitch
from chord_detection.chromagram import Chromagram
from chord_detection.dsp.wfir import wfir
from chord_detection.dsp.frame import frame_cutter
from chord_detection.dsp.lowpass import lowpass_filter
from chord_detection.periodicity import IterativeF0PeriodicityAnalysis
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
        peak_thresh=0.5,
        peak_min_dist=10,
        harmonic_multiples_elim=5,
    ):
        super().__init__(audio_path)
        self.frame_size = frame_size
        self.num_frames = math.ceil(self.x.shape[0] / self.frame_size)
        self.power = power
        self.channels = [
            229 * (10 ** ((zeta1 * c + zeta0) / 21.4) - 1) for c in range(channels)
        ]
        self.peak_thresh = peak_thresh
        self.peak_min_dist = peak_min_dist
        self.harmonic_multiples_elim = harmonic_multiples_elim
        self.periodicity_estimator = IterativeF0PeriodicityAnalysis(self.fs, self.frame_size)

    @staticmethod
    def display_name():
        return "Iterative F0 (Klapuri, Anssi)"

    @staticmethod
    def method_number():
        return 3

    def compute_pitches(self, display_plot_frame=-1):
        ycn = [None for _ in range(len(self.channels))]
        
        for i, fc in enumerate(self.channels):
            yc = _auditory_filterbank(self.x, self.fs, fc)
            yc = wfir(yc, self.fs, 12)  # dynamic level compression
            yc[yc < 0] = -yc[yc < 0]  # full-wave rectification
            yc = (
                yc + lowpass_filter(yc, self.fs, fc)
            ) / 2.0  # sum with low-pass filtered version of self at center-channel frequency

            ycn[i] = yc

        Yct = [
            [None for _ in range(len(self.channels))] for _ in range(self.num_frames)
        ]
        Ut = [None for _ in range(self.num_frames)]

        for channel, fc in enumerate(self.channels):
            for frame, yct in enumerate(frame_cutter(ycn[channel], self.frame_size)):
                # hamming windowed and zero-padded to 2x length
                yct = yct * scipy.signal.hamming(yct.shape[0])
                yct = numpy.concatenate((yct, numpy.zeros(yct.shape[0])))
                Yct[frame][channel] = yct.copy()

        shape = Yct[0][0].shape[0]
        for frame in range(self.num_frames):
            running_sum = numpy.zeros(shape)

            for channel, Yct_ in enumerate(Yct[frame]):
                running_sum += numpy.abs(numpy.fft.fft(Yct_)) ** self.power
            Ut[frame] = running_sum

        overall_chromagram = Chromagram()
        # periodicity estimate - iterative f0 cancellation/tau/salience loop
        for frame, Uk in enumerate(Ut):
            frame_chromagram, salience_plots = self.periodicity_estimator.compute(Uk)
            overall_chromagram += frame_chromagram

            if frame == display_plot_frame:
                _display_plots(self.clip_name, self.fs, self.frame_size, self.x, self.channels, ycn, Ut[frame], salience_plots)

        return overall_chromagram


def _display_plots(clip_name, fs, frame_size, x, channels, ytc, Ut, splots):
    fig1, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title(r"x[n], $y_c$[n], normalized - {0}".format(clip_name))
    ax1.set_xlabel("n (samples)")
    ax1.set_ylabel("amplitude")
    ax1.plot(
        numpy.arange(frame_size),
        x[: frame_size]/numpy.max(x),
        "b",
        alpha=0.75,
        linestyle="--",
        label="x[n]",
    )

    for i, x in enumerate(
        [random.randrange(0, len(channels)) for _ in range(6)]
    ):
        ax1.plot(
            numpy.arange(frame_size),
            ytc[x][: frame_size]/numpy.max(ytc[x][: frame_size]),
            color="C{0}".format(i),
            linestyle="--",
            alpha=0.5,
            label=r"$y_c$[n], $f_c$ = {0}".format(round(channels[x], 2)),
        )
    i += 1

    ax1.grid()
    ax1.legend(loc="upper right")

    ax2.set_title("Ut, bandwise summary spectrum")
    ax2.set_xlabel("fft bin")
    ax2.set_ylabel("amplitude")
    ax2.plot(
        numpy.arange(frame_size/2-1024),
        Ut[1024 : int(frame_size/2)],
        "b",
        alpha=0.75,
        linestyle="--",
        label="Ut",
    )

    max_ut = numpy.amax(Ut[1024 : int(frame_size/2)])

    (saliences, periods) = splots

    tau = int(1/periods[0])

    ax2.plot(
            tau,
            max_ut/2,
            'rx',
            label=r'$s(\hat{\tau})$ = ' + str(round(saliences[0], 2))
        )

    pitch = fs / periods[0]
    note = librosa.hz_to_note(pitch, octave=False)
    pitch = round(pitch, 2)

    ax2.text(
            tau,
            1.1*(max_ut/2),
            '{0}, {1}'.format(pitch, note)
    )

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

    shape = x_channels[0].shape[0]
    running_sum = numpy.zeros(shape)

    for xc in x_channels:
        running_sum += numpy.abs(numpy.fft.fft(xc)) ** k

    return running_sum[:int((shape-1)/2)]
