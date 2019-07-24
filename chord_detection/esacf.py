import numpy
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
from .multipitch import Multipitch, Chromagram
from .wfir import wfir
from .notes import freq_to_note, NOTE_NAMES
from collections import OrderedDict


class MultipitchESACF(Multipitch):
    def __init__(self, audio_path, ham_ms=46.4, k=0.67, n_notes=12, n_peaks_elim=5):
        super().__init__(audio_path)
        self.ham_samples = int(self.fs * ham_ms / 1000.0)
        self.k = k
        self.n_peaks_elim = n_peaks_elim
        self.n_notes = n_notes

    def compute_pitches(self):
        diff = len(self.x) - self.ham_samples
        self.x = self.x * numpy.concatenate(
            (scipy.signal.hamming(self.ham_samples), numpy.zeros(diff))
        )

        # then, the 12th-order warped linear prediction filter
        self.x = wfir(self.x, self.fs, 12)

        x_highpass = _highpass_filter(self.x.copy(), self.fs)
        x_highpass = numpy.clip(x_highpass, 0, None)  # half-wave rectification
        x_highpass = lowpass_filter(x_highpass, self.fs, 1000)  # paper wants it

        x_lowpass = lowpass_filter(self.x.copy(), self.fs, 1000)

        self.x_sacf = _sacf(x_lowpass, x_highpass)
        self.x_esacf = _esacf(self.x_sacf, self.n_peaks_elim)

        # the data that's actually interesting
        self.interesting = self.ham_samples
        peaks, _ = scipy.signal.find_peaks(self.x_esacf[: self.interesting])
        peak_prominence, _, _ = scipy.signal.peak_prominences(
            self.x_esacf[: self.interesting], peaks
        )

        max_peak_prominences = numpy.argpartition(peak_prominence, -self.n_notes)[
            -self.n_notes :
        ]
        normalized_peak_prominences = peak_prominence / numpy.max(peak_prominence)

        chromagram = Chromagram()

        for i in max_peak_prominences:
            pitch = self.fs / peaks[i]
            note = freq_to_note(pitch)
            chromagram[note] += normalized_peak_prominences[i]

        chromagram.normalize()
        return chromagram


def _sacf(x_low: numpy.ndarray, x_high: numpy.ndarray, k=None) -> numpy.ndarray:
    if not k:
        k = 0.67
    left = numpy.abs(numpy.fft.fft(x_low)) ** k
    right = numpy.abs(numpy.fft.fft(x_high)) ** k
    x2 = numpy.fft.ifft(left + right)
    return numpy.real(x2)


def _esacf(x2: numpy.ndarray, n_peaks=10) -> numpy.ndarray:
    """
    enhance the SACF with the following procedure
    clip to positive values, time stretch by n_peaks
    subtract original
    """
    x2tmp = x2.copy()

    for timescale in range(2, n_peaks + 1):
        x2tmp = numpy.clip(x2tmp, 0, None)
        x2stretched = librosa.effects.time_stretch(x2tmp, timescale)
        x2stretched = numpy.pad(
            x2stretched, (0, x2tmp.shape[0] - x2stretched.shape[0]), "constant"
        )
        x2tmp -= x2stretched
        x2tmp = numpy.clip(x2tmp, 0, None)

    return x2tmp


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
