from abc import ABCMeta, abstractmethod
from pathlib import Path
import librosa
from collections import OrderedDict

METHODS = OrderedDict()


class Multipitch(object):
    __metaclass__ = ABCMeta

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        method_num = cls.method_number()
        if method_num in METHODS.keys():
            raise ValueError(
                "Method number {0} already registered as {1} in {2}".format(
                    method_num, METHODS[method_num], METHODS
                )
            )
        METHODS[cls.method_number()] = cls

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

    @staticmethod
    @abstractmethod
    def display_name():
        raise ValueError("unimplemented")

    @staticmethod
    @abstractmethod
    def method_number():
        raise ValueError("unimplemented")
