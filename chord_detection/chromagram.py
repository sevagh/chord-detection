from collections import OrderedDict
from collections.abc import Sequence
import math
import numpy
import scipy


_note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class Chromagram(Sequence):
    def __init__(self):
        self.c = OrderedDict()
        for n in _note_names:
            self.c[n] = 0.0
        self.p = None
        super().__init__()

    def __getitem__(self, i):
        if type(i) == str:
            i = i.replace('♯', '#') # librosa-isms...
            return self.c[i]
        elif type(i) == int:
            return self.c[_note_names[i]]
        else:
            raise ValueError("this shouldn't happen")

    def __setitem__(self, i, item):
        if type(i) == str:
            self.c[i] = item
        elif type(i) == int:
            self.c[_note_names[i]] = item
        else:
            raise ValueError("this shouldn't happen")

    def __len__(self):
        return len(self.c)

    def __repr__(self):
        return self._pack()

    def __add__(self, other):
        for k in self.c.keys():
            self.c[k] += other.c[k]
        return self

    def key(self):
        return detect_key(numpy.asarray([v for v in self.c.values()]))

    def _pack(self):
        nc = _normalize(self.c)

        pack = [0 for _ in range(12)]

        for i, v in enumerate(nc.values()):
            pack[i] = int(round(v))

        return "".join([str(p_) for p_ in pack])


def _normalize(c: OrderedDict):
    c_ = c.copy()

    chromagram_min = min(c_.values())
    if chromagram_min != 0.0:
        for k in c_.keys():
            c_[k] = round(c_[k] / chromagram_min, 3)

    chromagram_max = max(c_.values())
    if chromagram_max > 9.0:
        for k in c_.keys():
            c_[k] *= 9.0 / chromagram_max

    return c_


"""
attribution:
    https://gist.github.com/bmcfee/1f66825cef2eb34c839b42dddbad49fd
    https://github.com/bmcfee
"""


def detect_key(X):
    if X.shape[0] != 12:
        raise ValueError(
            "input must be a chroma vector i.e. a numpy ndarray of shape (12,)"
        )
    # key_names = "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"
    X = scipy.stats.zscore(X)

    # Coefficients from Kumhansl and Schmuckler
    # as reported here: http://rnhart.net/articles/key-finding/
    major = numpy.asarray(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    major = scipy.stats.zscore(major)

    minor = numpy.asarray(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )
    minor = scipy.stats.zscore(minor)

    # Generate all rotations of major
    major = scipy.linalg.circulant(major)
    minor = scipy.linalg.circulant(minor)

    major = major.T.dot(X)
    minor = minor.T.dot(X)

    major_winner = int(numpy.argmax(major) + 0.5)
    minor_winner = int(numpy.argmax(minor) + 0.5)
    # essentia adds a 0.5? why
    # https://github.com/MTG/essentia/blob/master/src/algorithms/tonal/key.cpp#L370

    if major[major_winner] > minor[minor_winner]:
        return "{0}maj".format(_note_names[major_winner])
    elif major[major_winner] < minor[minor_winner]:
        return "{0}min".format(_note_names[minor_winner])
    else:
        if major_winner == minor_winner:
            return "{0}majmin".format(_note_names[major_winner])
        else:
            return "{0}maj OR {1}min".format(
                _note_names[major_winner], _note_names[minor_winner]
            )
