import unittest
import numpy
import matplotlib.pyplot as plt
from chord_detection import MultipitchESACF, MultipitchIterativeF0


class TestChordDetection(unittest.TestCase):
    def test_esacf_plots(self):
        esacf = MultipitchESACF("testclip.wav")
        esacf.compute_pitches()
        esacf.display_plots()

    def test_iterative_f0_plots(self):
        iterativef0 = MultipitchIterativeF0("testclip.wav")
        iterativef0.compute_pitches()
        iterativef0.compute_pitches()
