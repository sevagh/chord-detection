from math import log2, pow


A4 = 440
C0 = A4 * pow(2, -4.75)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note(freq, append_octave=False):
    h = round(12 * log2(freq / C0))
    octave = h // 12
    n = h % 12
    ret = NOTE_NAMES[n]
    if append_octave:
        ret += str(octave)
    return ret


def octave_frequencies():

