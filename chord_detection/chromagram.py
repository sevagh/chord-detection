from collections import OrderedDict
from collections.abc import Sequence
from .notes import NOTE_NAMES
import math
from pprint import pformat


class Chromagram(Sequence):
    def __init__(self):
        self.c = OrderedDict()
        for n in NOTE_NAMES:
            self.c[n] = 0.0
        self.p = None
        super().__init__()

    def __getitem__(self, i):
        if type(i) == str:
            return self.c[i]
        elif type(i) == int:
            return self.c[NOTE_NAMES[i]]
        else:
            raise ValueError("this shouldn't happen")

    def __setitem__(self, i, item):
        if type(i) == str:
            self.c[i] = item
        elif type(i) == int:
            self.c[NOTE_NAMES[i]] = item
        else:
            raise ValueError("this shouldn't happen")

    def __len__(self):
        return len(self.c)

    def __repr__(self):
        return pformat(self.c)

    def __add__(self, other):
        for k in self.c.keys():
            self.c[k] += other.c[k]
        return self

    def _normalize(self):
        chromagram_min = min(self.c.values())
        if chromagram_min == 0.0:
            return

        for k in self.c.keys():
            self.c[k] = round(self.c[k] / chromagram_min, 3)

    def pack(self):
        self._normalize()
        chromagram_max = max(self.c.values())
        if chromagram_max > 9.0:
            for k in self.c.keys():
                self.c[k] *= 9.0 / chromagram_max

        pack = [0 for _ in range(12)]

        for i, v in enumerate(self.c.values()):
            pack[i] = int(round(v))

        return "".join([str(p_) for p_ in pack])
