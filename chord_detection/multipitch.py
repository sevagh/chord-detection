from abc import ABC, abstractmethod
from pathlib import Path
import librosa


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
