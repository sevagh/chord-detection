from abc import ABC, abstractmethod
from pathlib import Path
import librosa
from collections import OrderedDict
from collections.abc import Sequence
from .notes import NOTE_NAMES
from pprint import pformat


class Multipitch(ABC):
    @abstractmethod
    def __init__(self, audio_path):
        x, self.fs = librosa.load(audio_path)
        if len(x.shape) == 2 and x.shape[0] == 2:
            self.x = x[0::2] / 2.0 + x[1::2] / 2.0
        else:
            self.x = x
        self.clip_name = Path(audio_path).name

    @abstractmethod
    def compute_pitches(self):
        pass

    @abstractmethod
    def display_plots(self):
        pass

    @abstractmethod
    def display_name(self):
        pass


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
