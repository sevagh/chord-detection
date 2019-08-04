import numpy
import scipy
import scipy.signal
import librosa
import typing
import peakutils
import matplotlib.pyplot as plt
from ..multipitch import Multipitch
from ..chromagram import Chromagram
from ..dsp.wfir import wfir
from ..music.notes import freq_to_note, NOTE_NAMES
from ..dsp.frame import frame_cutter
from collections import OrderedDict


class MultipitchESACF(Multipitch):
    def __init__(
        self,
        audio_path,
        ham_ms=46.4,
        k=0.67,
        n_peaks_elim=6,
        peak_thresh=0.1,
        peak_min_dist=10,
    ):
        super().__init__(audio_path)
        self.ham_samples = int(self.fs * ham_ms / 1000.0)
        self.k = k
        self.n_peaks_elim = n_peaks_elim
        self.peak_thresh = peak_thresh
        self.peak_min_dist = peak_min_dist

    @staticmethod
    def display_name():
        return "ESACF (Tolonen, Karjalainen)"

    @staticmethod
    def method_number():
        return 1

    def compute_pitches(self, display_plot_frame=-1):
        overall_chromagram = Chromagram()

        for frame, x_frame in enumerate(frame_cutter(self.x, self.ham_samples)):
            x = wfir(x_frame, self.fs, 12)

            x_hi = _highpass_filter(x, self.fs)
            x_hi = numpy.clip(x_hi, 0, None)  # half-wave rectification
            x_hi = lowpass_filter(x_hi, self.fs, 1000)  # paper wants it

            x_lo = lowpass_filter(x, self.fs, 1000)

            x_sacf = _sacf([x_lo, x_hi])
            x_esacf, harmonic_elim_plots = _esacf(x_sacf, self.n_peaks_elim, True)

            peak_indices = peakutils.indexes(
                x_esacf, thres=self.peak_thresh, min_dist=self.peak_min_dist
            )

            peak_indices_interp = peakutils.interpolate(
                numpy.arange(x_esacf.shape[0]), x_esacf, ind=peak_indices
            )

            chromagram = Chromagram()
            for i, tau in enumerate(peak_indices_interp):
                pitch = self.fs / tau
                note = freq_to_note(pitch)
                chromagram[note] += x_esacf[peak_indices[i]]
            overall_chromagram += chromagram

            if frame == display_plot_frame:
                _display_plots(
                    self.clip_name,
                    self.fs,
                    self.ham_samples,
                    frame,
                    x,
                    x_lo,
                    x_hi,
                    x_sacf,
                    x_esacf,
                    harmonic_elim_plots,
                    peak_indices,
                    peak_indices_interp,
                )

        return overall_chromagram.pack()


def _sacf(x_channels: typing.List[numpy.ndarray], k=None) -> numpy.ndarray:
    # k is same as p (power) in the Klapuri/Ansi paper, method 3
    if not k:
        k = 0.67

    shape = x_channels[0].shape[0]

    running_sum = numpy.zeros(shape)

    for xc in x_channels:
        running_sum += numpy.abs(numpy.fft.fft(xc)) ** k

    return numpy.real(numpy.fft.ifft(running_sum))[:int((shape-1)/2)]


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


def _display_plots(
    clip_name,
    fs,
    frame_size,
    frame,
    x,
    x_lo,
    x_hi,
    x_sacf,
    x_esacf,
    harmonic_elim_plots,
    peak_indices,
    peak_indices_interp,
):
    samples = numpy.arange(frame_size)

    fig1, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title("{0} - x[n], frame {1}".format(clip_name, frame))
    ax1.set_xlabel("n (samples)")
    ax1.set_ylabel("amplitude")
    ax1.plot(samples, x, "b", alpha=0.5, label="x[n]")
    ax1.plot(samples, x_lo, "g", alpha=0.5, linestyle="--", label="x[n] lo")
    ax1.plot(samples, x_hi, "r", alpha=0.5, linestyle=":", label="x[n] hi")
    ax1.grid()
    ax1.legend(loc="upper right")

    ax2.set_title("SACF, ESACF")
    ax2.set_xlabel("n (samples)")
    ax2.set_ylabel("normalized amplitude")

    i = 0
    for i, h in enumerate(harmonic_elim_plots):
        h_norm = h / numpy.max(h)
        ax2.plot(
            samples,
            numpy.concatenate((h_norm, numpy.zeros(samples.shape[0] - h.shape[0]))),
            "C{0}".format(i),
            alpha=0.1,
            label="time stretch {0}".format(2 + i),
        )
    i += 1
    sacf_norm = x_sacf / numpy.max(x_sacf)
    ax2.plot(
        samples,
        numpy.concatenate(
            (sacf_norm, numpy.zeros(samples.shape[0] - sacf_norm.shape[0]))
        ),
        "C{0}".format(i),
        linestyle="--",
        alpha=0.5,
        label="sacf",
    )
    esacf_norm = x_esacf / numpy.max(x_esacf)
    i += 1
    ax2.plot(
        samples,
        numpy.concatenate(
            (esacf_norm, numpy.zeros(samples.shape[0] - sacf_norm.shape[0]))
        ),
        "C{0}".format(i),
        linestyle=":",
        alpha=0.5,
        label="esacf",
    )
    scatter_peaks = esacf_norm[peak_indices]
    for i, ind in enumerate(peak_indices_interp):
        pitch = round(fs / ind, 2)
        text = "{0}, {1}".format(pitch, freq_to_note(pitch))
        x = peak_indices_interp[i]
        y = scatter_peaks[i]
        ax2.plot(x, y, "rx")
        ax2.text(x, y, text)

    ax2.grid()
    ax2.legend(loc="upper right")

    plt.show()
