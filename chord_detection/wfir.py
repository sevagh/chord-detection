import numpy
import scipy
import scipy.signal


def _bark_warp_coef(fs):
        return 1.0674 * numpy.sqrt((2.0 / numpy.pi) * numpy.arctan(0.06583 * fs / 1000.0)) - 0.1916
    
    
def _warped_remez_coefs(fs, order):
    l = 20
    r = min(20000, fs/2 - 1)
    t = 1
      
    c = scipy.signal.remez(order+1, [0, l-t, l, r, r+t, 0.5*fs], [0, 1, 0], fs=fs)
    return c.tolist()

    
# see: https://sevagh.github.io/warped-linear-prediction/
def wfir(x: numpy.ndarray, fs: float, order: int) -> numpy.ndarray:
    a = _bark_warp_coef(fs)

    B = [-a.conjugate(), 1]
    A = [1, -a]
    ys = [0] * order

    ys[0] = scipy.signal.lfilter(B, A, x)
    for i in range(1, len(ys)):
        ys[i] = scipy.signal.lfilter(B, A, ys[i - 1])
        
    c = _warped_remez_coefs(fs, order)

    x_hat = c[0] * x
    for i in range(order):
        x_hat += c[i+1] * ys[i]

    r = x - x_hat
    return r
