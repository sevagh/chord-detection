import unittest
import numpy
import os
from chord_detection import (
    MultipitchESACF,
    MultipitchIterativeF0,
    MultipitchHarmonicEnergy,
    MultipitchPrimeMultiF0,
)
from chord_detection.notes import gen_octave
import soundfile
from tempfile import TemporaryDirectory


TESTCASES = {
    "tests/test_1_note_Csharp3.wav": "010000000000",
    "tests/test_1_note_E4.wav": "000010000000",
    "tests/test_2_notes_E2_F3.wav": "000011000000",
    "tests/test_2_notes_G3_Asharp4.wav": "000000010010",
    "tests/test_3_notes_G2_B2_G#3.wav": "000000011001",
}


class TestChordDetection(unittest.TestCase):
    def test_all(self):
        for test_clip, expected_result in TESTCASES.items():
            compute_objs = [
                MultipitchESACF(test_clip),
                MultipitchHarmonicEnergy(test_clip),
                MultipitchIterativeF0(test_clip),
                MultipitchPrimeMultiF0(test_clip),
            ]
            for c in compute_objs:
                ret = c.compute_pitches().pack()
                print(
                    "{0}\n{1}\n{2} expected\n{3} actual\n".format(
                        c.display_name(), test_clip, expected_result, ret
                    )
                )
                c.display_plots()
