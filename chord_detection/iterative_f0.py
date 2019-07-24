import numpy
import math
import scipy
import scipy.signal
import scipy.fftpack
import librosa
from numba import jit, njit
from .multipitch import Multipitch, Chromagram
from .notes import freq_to_note, gen_octave, NOTE_NAMES
from .wfir import wfir
from .esacf import lowpass_filter
from collections import OrderedDict


class MultipitchIterativeF0(Multipitch):
    def __init__(self, audio_path, frame_size=8192, power=0.67, channels=70, epsilon0=2.3, epsilon1=0.39, num_harmonic=2, num_octave=2, num_bins=2):
        super().__init__(audio_path)
        self.frame_size = frame_size
        self.power = power
        num_frames = float(len(self.x)) / float(frame_size)
        num_frames = int(math.ceil(num_frames))
        self.num_frames = num_frames
        pad = int(num_frames * frame_size - len(self.x))
        self.x = numpy.concatenate((self.x, numpy.zeros(pad)))
        self.channels = [229*(10**((epsilon1*c + epsilon0)/21.4) - 1) for c in range(channels)]
        # salience calculation apparatus
        # borrowed from method2
        self.num_harmonic = num_harmonic
        self.num_octave = num_octave
        self.num_bins = num_bins

    @jit
    def compute_pitches(self):
        Utc = [[None] * len(self.channels)]*self.num_frames

        for i, fc in enumerate(self.channels):
            yc = _auditory_filterbank(self.x, self.fs, fc)

            yc = wfir(yc, self.fs, 12)  # dynamic level compression
            yc[yc < 0] = -yc[yc < 0]  # full-wave rectification

            yc = (
                yc + lowpass_filter(yc, self.fs, fc)
            ) / 2.0  # sum with low-pass filtered version of self at center-channel frequency

            for j, yc_ in enumerate(numpy.split(yc, self.num_frames)):  # split yc into frames
                if len(yc_) < self.frame_size:
                    yc_ = numpy.concatenate((yc_, numpy.zeros(self.frame_size - len(yc_))))

                yc_ *= scipy.signal.hamming(self.frame_size)
                yc_ = numpy.concatenate(
                    (yc_, numpy.zeros(self.frame_size))
                )

                #Y = scipy.fftpack.fft(yc_)
                #Y = numpy.fft.fft(yc_)
                Y = librosa.core.stft(

                Utc[j][i] = numpy.absolute(Y) ** self.power

        Ut = [sum(x) for x in Utc]

        # salience calculation in section D looks similar to the chroma vector from method 2, "Real-Time Chord Detection for Live Performance", equation (1)
        salience_t = []

        # first C = C3 aka 130.81 Hz
        # for memoization, apparently
        notes = list(gen_octave(130.81))
        divisor_ratio = (self.fs / 4.0) / float((Ut[0]).shape[0])

        for U in Ut:
            salience = [0.0] * 12
            for n in range(12):
                chroma_sum = 0.0
                for octave in range(1, self.num_octave + 1):
                    note_sum = 0.0
                    for harmonic in range(1, self.num_harmonic + 1):
                        U_max = float("-inf")  # sentinel

                        k_prime = numpy.round(
                            (notes[n] * octave * harmonic) / divisor_ratio
                        )
                        k0 = int(k_prime - self.num_bins * harmonic)
                        k1 = int(k_prime + self.num_bins * harmonic)

                        for k in range(k0, k1):
                            U_max = max(U[k], U_max)

                        note_sum += U_max
                    chroma_sum += note_sum
                salience[n] += chroma_sum
            salience_t.append(salience.copy())

        chromagram = Chromagram()

        for n in range(12):
            chromagram[n] = 0.0
            for s in salience_t:
                chromagram[n] += s[n]
            chromagram[n] /= len(salience_t)

        chromagram.normalize()
        return chromagram

def _auditory_filterbank(x, fc, fs):
    J = 4

    # bc3db = -3/J dB
    A = numpy.exp(-(3 / J) * numpy.pi / (fs * numpy.sqrt(2 ** (1 / J) - 1)))

    cos_theta1 = (1+A*A)/(2*A) * numpy.cos(2*numpy.pi*fc/fs)
    cos_theta2 = (2*A)/(1+A*A) * numpy.cos(2*numpy.pi*fc/fs)
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
