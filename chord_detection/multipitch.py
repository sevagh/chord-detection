from abc import ABC, abstractmethod
from pathlib import Path
import librosa


class Multipitch(ABC):
    @abstractmethod
    def __init__(self, audio_path):
        self.x, self.fs = librosa.load(audio_path)
        self.clip_name = Path(audio_path).name

    @abstractmethod
    def compute_pitches(self):
        pass

    @abstractmethod
    def display_plots(self):
        pass
