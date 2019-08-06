import unittest
import numpy
import os
from chord_detection import detect_key
import soundfile
from tempfile import TemporaryDirectory


TESTCASES = {
    "Cmaj": numpy.asarray(
        [
            100.0,  # C-E-G, c major pitches
            0.0,
            0.0,
            0.0,
            100.0,  # E
            0.0,
            0.0,
            100.0,  # G
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ),
    "Cmin": numpy.asarray(
        [
            50.0,  # C-D-Eb-G, c minor pitches
            0.0,
            50.0,  # D
            50.0,  # D#/Eb
            0.0,
            0.0,
            0.0,
            10.0,  # G
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ),
    "G#maj": numpy.asarray(
        [
            0.0,
            10.0,  # C#
            0.0,
            10.0,  # D#
            0.0,
            0.0,
            0.0,
            0.0,
            10.0,  # G#
            0.0,
            10.0,  # A#
            0.0,
        ]
    ),
}


class TestKeyDetection(unittest.TestCase):
    def test_krumhansl_schmuckler_key_detection(self):
        for expected_key, X in TESTCASES.items():
            self.assertEqual(detect_key(X), expected_key)
