#!/usr/bin/env python3.7

import unittest
import numpy
import os
import librosa
from chord_detection.notes import gen_octave
import soundfile
import sys
from tempfile import TemporaryDirectory


def main():
    dest = "test_1_note_Csharp3.wav"
    tone = librosa.tone(138.59, sr=22050, length=44100)
    soundfile.write(dest, tone, 22050)
    print("Created {0} with note C#3".format(dest))

    dest = "test_1_note_E4.wav"
    tone = librosa.tone(329.63, sr=22050, length=44100)
    soundfile.write(dest, tone, 22050)
    print("Created {0} with note E4".format(dest))

    dest = "test_2_notes_E2_F3.wav"
    tone = numpy.zeros(44100)
    tone += librosa.tone(82.41, sr=22050, length=44100)
    tone += librosa.tone(174.61, sr=22050, length=44100)
    soundfile.write(dest, tone, 22050)
    print("Created {0} with notes E2, F3".format(dest))

    dest = "test_2_notes_G3_Asharp4.wav"
    tone = numpy.zeros(44100)
    tone += librosa.tone(196, sr=22050, length=44100)
    tone += librosa.tone(466.16, sr=22050, length=44100)
    soundfile.write(dest, tone, 22050)
    print("Created {0} with notes G3, A#4".format(dest))

    dest = "test_3_notes_G2_B2_G#3.wav"
    tone = numpy.zeros(44100)
    tone += librosa.tone(98, sr=22050, length=44100)
    tone += librosa.tone(123.47, sr=22050, length=44100)
    tone += librosa.tone(207.65, sr=22050, length=44100)
    soundfile.write(dest, tone, 22050)
    print("Created {0} with notes G2, B2, G#3".format(dest))

    return 0


if __name__ == "__main__":
    sys.exit(main())
