import numpy
import math
import random
import scipy
import scipy.signal
import librosa
import typing
import peakutils
from matplotlib import mlab
import matplotlib.pyplot as plt
from ..multipitch import Multipitch
from ..chromagram import Chromagram
from ..dsp.wfir import wfir
from ..music.notes import freq_to_note, gen_octave, NOTE_NAMES
from ..dsp.frame import frame_cutter
from collections import OrderedDict


class MultipitchPrimeMultiF0(Multipitch):
    def __init__(
        self,
        audio_path,
        num_harmonic=1,
        num_octave=2,
        harmonic_multiples_elim=5,
        harmonic_elim_runs=2,
    ):
        super().__init__(audio_path)
        self.num_harmonic = num_harmonic
        self.num_octave = num_octave
        self.harmonic_elim_runs = harmonic_elim_runs
        self.harmonic_multiples_elim = harmonic_multiples_elim

    @staticmethod
    def display_name():
        return "Prime-multiF0 (Camacho, Kaver-Oreamuno)"

    @staticmethod
    def method_number():
        return 4

    def compute_pitches(self):
        overall_chromagram = Chromagram()

        # first C = C3 aka 130.81 Hz
        notes = list(gen_octave(130.81))

        self.specgram_to_plot = []

        for n in range(12):
            for octave in range(1, self.num_octave + 1):
                for harmonic in range(1, self.num_harmonic + 1):
                    f_candidate = notes[n] * octave * harmonic
                    window_size = int((8 / f_candidate) * self.fs)

                    chromagram = Chromagram()
                    for i, x_t in enumerate(frame_cutter(self.x, window_size)):
                        real_window_size = max(x_t.shape[0], window_size)
                        window = numpy.hanning(real_window_size)
                        s, f = mlab.magnitude_spectrum(x_t, Fs=self.fs, window=window)
                        s[s < 0] = 0.0  # clip
                        might_append_1 = s.copy()
                        might_append_2 = []

                        for _ in range(self.harmonic_elim_runs):
                            max_freq_idx = s.argmax(axis=0)
                            max_f = f[max_freq_idx]
                            try:
                                note = freq_to_note(max_f)
                                chromagram[note] += s[max_freq_idx]
                                might_append_2.append((max_freq_idx, max_f, note))
                            except ValueError:
                                continue
                            eliminated = []
                            for harmonic_index_multiple in range(
                                1, self.harmonic_multiples_elim
                            ):
                                elim_freq = harmonic_index_multiple * max_f
                                elim_index = numpy.where(f == elim_freq)
                                s[elim_index] -= s[elim_index]
                        might_append_3 = s.copy()

                        if not random.getrandbits(1):
                            self.specgram_to_plot.append(
                                (might_append_1, might_append_2, might_append_3)
                            )
                    overall_chromagram += chromagram

        return overall_chromagram.pack()

    def display_plots(self):
        fig1, (ax1, ax2) = plt.subplots(2, 1)

        ax1.set_title("x[n] - {0}".format(self.clip_name))
        ax1.set_xlabel("n (samples)")
        ax1.set_ylabel("amplitude")
        ax1.plot(numpy.arange(self.x.shape[0]), self.x, "b", alpha=0.5, label="x[n]")
        ax1.grid()
        ax1.legend(loc="upper right")

        (s, notes, s_post) = random.choice(self.specgram_to_plot)

        ax2.set_title("S (specgram)".format(self.clip_name))
        ax2.set_xlabel("frequency bins")
        ax2.set_ylabel("magnitude")
        ax2.plot(numpy.arange(s.shape[0]), s, "b", alpha=0.5, label="S")
        for (freq_idx, freq, note) in notes:
            ax2.plot(freq_idx, s[freq_idx], "ro")
            ax2.text(freq_idx, s[freq_idx], "{0}, {1}".format(round(freq, 2), note))
        ax2.plot(
            numpy.arange(s_post.shape[0]),
            s_post,
            "g",
            alpha=0.5,
            label="S' (f0 candidates eliminated)",
        )
        ax2.grid()
        ax2.legend(loc="upper right")

        plt.show()
