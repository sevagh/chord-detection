import numpy
import scipy
import scipy.signal
import librosa
import typing
import peakutils
import matplotlib.pyplot as plt
from .multipitch import Multipitch
from .chromagram import Chromagram
from .wfir import wfir
from .notes import freq_to_note, NOTE_NAMES
from collections import OrderedDict


class MultipitchESACF(Multipitch):
    def __init__(
        self,
        audio_path,
        ham_ms=46.4,
        k=0.67,
        n_peaks_elim=6,
        peak_thresh=0.5,
        peak_min_dist=10,
    ):
        super().__init__(audio_path)
        self.ham_samples = int(self.fs * ham_ms / 1000.0)
        self.k = k
        self.n_peaks_elim = n_peaks_elim
        self.peak_thresh = peak_thresh
        self.peak_min_dist = peak_min_dist

    def display_name(self):
        return "ESACF (Tolonen, Karjalainen)"

    def compute_pitches(self):
        diff = len(self.x) - self.ham_samples
        self.x = self.x * numpy.concatenate(
            (scipy.signal.hamming(self.ham_samples), numpy.zeros(diff))
        )

        # then, the 12th-order warped linear prediction filter
        self.x = wfir(self.x, self.fs, 12)

        self.x_hi = _highpass_filter(self.x.copy(), self.fs)
        self.x_hi = numpy.clip(self.x_hi, 0, None)  # half-wave rectification
        self.x_hi = lowpass_filter(self.x_hi, self.fs, 1000)  # paper wants it

        self.x_lo = lowpass_filter(self.x.copy(), self.fs, 1000)

        self.x_sacf = _sacf([self.x_lo, self.x_hi])
        self.x_esacf, self.harmonic_elim_plots = _esacf(
            self.x_sacf, self.n_peaks_elim, True
        )

        # the data that's actually interesting
        self.interesting = self.ham_samples

        x_esacf_ = self.x_esacf[: self.interesting]
        self.peak_indices = peakutils.indexes(
            x_esacf_, thres=self.peak_thresh, min_dist=self.peak_min_dist
        )

        self.peak_indices_interp = peakutils.interpolate(
            numpy.arange(self.interesting), x_esacf_, ind=self.peak_indices
        )

        chromagram = Chromagram()
        for i, tau in enumerate(self.peak_indices_interp):
            pitch = self.fs / tau
            note = freq_to_note(pitch)
            chromagram[note] += x_esacf_[self.peak_indices[i]]
        chromagram.normalize()
        return chromagram

    def display_plots(self):
        samples = numpy.arange(self.interesting)

        fig1, (ax1, ax2) = plt.subplots(2, 1)

        ax1.set_title("x[n] - {0}".format(self.clip_name))
        ax1.set_xlabel("n (samples)")
        ax1.set_ylabel("amplitude")
        ax1.plot(samples, self.x[: self.interesting], "b", alpha=0.5, label="x[n]")
        ax1.plot(
            samples,
            self.x_lo[: self.interesting],
            "g",
            alpha=0.5,
            linestyle="--",
            label="x[n] lo",
        )
        ax1.plot(
            samples,
            self.x_hi[: self.interesting],
            "r",
            alpha=0.5,
            linestyle=":",
            label="x[n] hi",
        )
        ax1.grid()
        ax1.legend(loc="upper right")

        ax2.set_title("SACF, ESACF")
        ax2.set_xlabel("n (samples)")
        ax2.set_ylabel("normalized amplitude")

        i = 0
        for i, h in enumerate(self.harmonic_elim_plots):
            h_norm = h[: self.interesting] / numpy.max(h[: self.interesting])
            ax2.plot(
                samples,
                h_norm,
                "C{0}".format(i),
                alpha=0.1,
                label="time stretch {0}".format(2 + i),
            )
        i += 1
        sacf_norm = self.x_sacf[: self.interesting] / numpy.max(
            self.x_sacf[: self.interesting]
        )
        ax2.plot(
            samples,
            sacf_norm,
            "C{0}".format(i),
            linestyle="--",
            alpha=0.75,
            label="sacf",
        )
        esacf_norm = self.x_esacf[: self.interesting] / numpy.max(
            self.x_esacf[: self.interesting]
        )
        i += 1
        ax2.plot(
            samples,
            esacf_norm,
            "C{0}".format(i),
            linestyle=":",
            alpha=0.75,
            label="esacf",
        )
        scatter_peaks = esacf_norm[self.peak_indices]
        for i, ind in enumerate(self.peak_indices_interp):
            pitch = round(self.fs / ind, 2)
            text = "{0}, {1}".format(pitch, freq_to_note(pitch))
            x = self.peak_indices_interp[i]
            y = scatter_peaks[i]
            ax2.plot(x, y, "rx")
            ax2.text(x, y, text)

        ax2.grid()
        ax2.legend(loc="lower left")

        plt.show()


def _sacf(x_channels: typing.List[numpy.ndarray], k=None) -> numpy.ndarray:
    # k is same as p (power) in the Klapuri/Ansi paper, method 3
    if not k:
        k = 0.67

    running_sum = numpy.zeros(x_channels[0].shape[0])

    for xc in x_channels:
        running_sum += numpy.abs(numpy.fft.fft(xc)) ** k

    return numpy.real(numpy.fft.ifft(running_sum))


def _esacf(
    x2: numpy.ndarray, n_peaks: int, ret_plots: bool
) -> typing.Tuple[numpy.ndarray, typing.List[numpy.ndarray]]:
    """
    enhance the SACF with the following procedure
    clip to positive values, time stretch by n_peaks
    subtract original
    """
    x2tmp = x2.copy()
    to_plot = []

    for timescale in range(2, n_peaks + 1):
        x2tmp = numpy.clip(x2tmp, 0, None)
        x2stretched = librosa.effects.time_stretch(x2tmp, timescale).copy()

        x2stretched.resize(x2tmp.shape)
        if ret_plots:
            to_plot.append(x2stretched)
        x2tmp -= x2stretched
        x2tmp = numpy.clip(x2tmp, 0, None)

    return x2tmp, to_plot


def _highpass_filter(x: numpy.ndarray, fs: float) -> numpy.ndarray:
    b, a = scipy.signal.butter(2, [1000 / (fs / 2)], btype="high")
    return scipy.signal.lfilter(b, a, x)


"""
Paper says:
    The lowpass block also includes a highpass rolloff with 12 dB/octave below 70 Hz.

    Still TODO?
"""


def lowpass_filter(x: numpy.ndarray, fs: float, band: float) -> numpy.ndarray:
    b, a = scipy.signal.butter(2, [band / (fs / 2)], btype="low")
    return scipy.signal.lfilter(b, a, x)
