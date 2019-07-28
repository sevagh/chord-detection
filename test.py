import unittest
import numpy
import os
from chord_detection import (
    MultipitchESACF,
    MultipitchIterativeF0,
    MultipitchHarmonicEnergy,
    MultipitchPrimeMultiF0,
)
import librosa
from chord_detection.notes import gen_octave
import soundfile
from tempfile import TemporaryDirectory


class TestChordDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # generate a tone with all 12 "notes"
        cls.tmpdir = TemporaryDirectory(suffix="chord_detection_unit_tests")
        dest = os.path.join(cls.tmpdir.name, "test_clip.wav")
        tone = numpy.zeros(44100)
        freqs = []
        for frequency in gen_octave(c_initial_hz=261.63):
            tone += librosa.tone(frequency, sr=22050, length=44100)
            freqs.append(str(round(frequency, 3)))
        print("Generated a tone with frequencies {0}".format(",".join(freqs)))
        soundfile.write(dest, tone, 22050)
        cls.test_wav_path = dest

    def test_esacf(self):
        esacf = MultipitchESACF(self.test_wav_path)
        print(esacf.display_name())
        ret = esacf.compute_pitches().pack()
        esacf.display_plots()
        self.assertEqual(ret, "111111111111")

    def test_harmonic_energy(self):
        harmen = MultipitchHarmonicEnergy(self.test_wav_path)
        print(harmen.display_name())
        ret = harmen.compute_pitches().pack()
        harmen.display_plots()
        self.assertEqual(ret, "111111111111")

    def test_iterative_f0(self):
        iterativef0 = MultipitchIterativeF0(self.test_wav_path)
        print(iterativef0.display_name())
        ret = iterativef0.compute_pitches().pack()
        iterativef0.display_plots()
        self.assertEqual(ret, "111111111111")

    def test_prime_multif0(self):
        primef0 = MultipitchPrimeMultiF0(self.test_wav_path)
        print(primef0.display_name())
        cgram = primef0.compute_pitches()
        self.assertEqual(cgram.pack(), "111111111111")
        print(cgram)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()
