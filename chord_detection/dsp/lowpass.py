import scipy
import numpy
import scipy.signal


def lowpass_filter(x: numpy.ndarray, fs: float, band: float) -> numpy.ndarray:
    b, a = scipy.signal.butter(2, [band / (fs / 2)], btype="low")
    return scipy.signal.lfilter(b, a, x)
