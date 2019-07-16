import numpy
import scipy
import scipy.signal
import librosa
from .wfir import wfir


def multipitch_esacf(audio_file_path: str):
    x, fs = librosa.load(audio_file_path)

    # first, the 12th-order warped linear prediction filter
    x = wfir(x, fs, 12)

    x_highpass = highpass_filter(x.copy(), fs)
    x_highpass = numpy.clip(x_highpass, 0, None) # half-wave rectification
    x_highpass = lowpass_filter(x_highpass, fs) # paper wants it

    x_lowpass = lowpass_filter(x.copy(), fs)

    x_sacf = sacf(x_lowpass, x_highpass)
    x_esacf = esacf(x_sacf)

    return x, x_sacf, x_esacf


def sacf(x_low: numpy.ndarray, x_high: numpy.ndarray) -> numpy.ndarray:
    k = 0.67
    left = numpy.abs(numpy.fft.fft(x_low))**k
    right = numpy.abs(numpy.fft.fft(x_high))**k
    x2 = numpy.fft.ifft(left + right)
    return numpy.real(x2)


def esacf(x2: numpy.ndarray, n_peaks=2) -> numpy.ndarray:
    '''
    enhance the SACF with the following procedure
    clip to positive values, time stretch by n_peaks
    subtract original
    '''
    x2tmp = x2.copy()

    for timescale in range(2, n_peaks+1):
        x2tmp = numpy.clip(x2tmp, 0, None)
        x2stretched = librosa.effects.time_stretch(x2tmp, timescale)
        x2stretched = numpy.pad(x2stretched, (0, x2tmp.shape[0]-x2stretched.shape[0]), 'constant')
        x2tmp -= x2stretched
        x2tmp = numpy.clip(x2tmp, 0, None)

    return x2tmp


def highpass_filter(x: numpy.ndarray, fs: float) -> numpy.ndarray:
    b, a = scipy.signal.butter(2, [1000/(fs/2)], btype='high')
    return scipy.signal.lfilter(b, a, x)


'''
Paper says:
    The lowpass block also includes a highpass rolloff with 12 dB/octave below 70 Hz.

    Still TODO
'''
def lowpass_filter(x: numpy.ndarray, fs: float) -> numpy.ndarray:
    b, a = scipy.signal.butter(2, [1000/(fs/2)], btype='low')
    return scipy.signal.lfilter(b, a, x)
