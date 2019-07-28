from collections import OrderedDict
from collections.abc import Sequence
from .notes import NOTE_NAMES
from pprint import pformat


class Chromagram(Sequence):
    def __init__(self):
        self.c = OrderedDict()
        for n in NOTE_NAMES:
            self.c[n] = 0.0
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

    def normalize(self):
        chromagram_max = max(self.c.values())
        if chromagram_max == 0.0:
            return

        for k in self.c.keys():
            self.c[k] = round(self.c[k] / chromagram_max, 3)

    def pack(self):
        self.normalize()

        final_chromagram = ""
        for v in self.c.values():
            bit_out = 0
            if v >= 0.5:
                bit_out = 1
            final_chromagram += str(bit_out)

        return final_chromagram
