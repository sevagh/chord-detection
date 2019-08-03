import numpy
import math
import random
import typing
import scipy
import scipy.signal
import scipy.fftpack
import peakutils
import librosa
import matplotlib.pyplot as plt
from numba import njit, jit
from ..multipitch import Multipitch
from ..chromagram import Chromagram
from ..music.notes import freq_to_note, gen_octave, NOTE_NAMES
from ..dsp.wfir import wfir
from ..tolonen_karjalainen.esacf import lowpass_filter
from ..dsp.frame import frame_cutter
from collections import OrderedDict


class MultipitchIterativeF0(Multipitch):
    def __init__(
        self,
        audio_path,
        frame_size=8192,
        power=1.0,
        channels=70,
        zeta0=2.3,
        zeta1=0.39,
        epsilon1=20,
        epsilon2=320,
        peak_thresh=0.5,
        peak_min_dist=10,
        harmonic_multiples_elim=5,
        M=20,
        tau_prec=1,
    ):
        super().__init__(audio_path)
        self.frame_size = frame_size
        self.num_frames = math.ceil(self.x.shape[0] / self.frame_size)
        self.power = power
        self.channels = [
            229 * (10 ** ((zeta1 * c + zeta0) / 21.4) - 1) for c in range(channels)
        ]
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.peak_thresh = peak_thresh
        self.peak_min_dist = peak_min_dist
        self.harmonic_multiples_elim = harmonic_multiples_elim
        self.M = M
        self.tau_prec = tau_prec

    @staticmethod
    def display_name():
        return "Iterative F0 (Klapuri, Anssi)"

    @staticmethod
    def method_number():
        return 3

    def compute_pitches(self, display_plot_frame=-1):
        ycn = [None for _ in range(len(self.channels))]
        
        for i, fc in enumerate(self.channels):
            yc = _auditory_filterbank(self.x, self.fs, fc)
            yc = wfir(yc, self.fs, 12)  # dynamic level compression
            yc[yc < 0] = -yc[yc < 0]  # full-wave rectification
            yc = (
                yc + lowpass_filter(yc, self.fs, fc)
            ) / 2.0  # sum with low-pass filtered version of self at center-channel frequency

            ycn[i] = yc

        Yct = [
            [None for _ in range(len(self.channels))] for _ in range(self.num_frames)
        ]
        Ut = [None for _ in range(self.num_frames)]

        for channel, fc in enumerate(self.channels):
            for frame, yct in enumerate(frame_cutter(ycn[channel], self.frame_size)):
                # hamming windowed and zero-padded to 2x length
                yct = yct * scipy.signal.hamming(yct.shape[0])
                yct = numpy.concatenate((yct, numpy.zeros(yct.shape[0])))
                Yct[frame][channel] = yct.copy()

        shape = Yct[0][0].shape[0]
        for frame in range(self.num_frames):
            running_sum = numpy.zeros(shape)

            for channel, Yct_ in enumerate(Yct[frame]):
                running_sum += numpy.abs(numpy.fft.fft(Yct_)) ** self.power
            Ut[frame] = running_sum

        # periodicity estimate - iterative f0 cancellation/tau/salience loop

        return None


def _display_plots(self):
    fig1, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title("x[n] - {0}".format(self.clip_name))
    ax1.set_xlabel("n (samples)")
    ax1.set_ylabel("amplitude")
    ax1.plot(
        numpy.arange(self.frame_size),
        self.x[: self.frame_size],
        "b",
        alpha=0.75,
        linestyle="--",
        label="x[n]",
    )

    ax1.grid()
    ax1.legend(loc="upper right")

    ax2.set_title(r"$y_c$[n], auditory filterbank")
    ax2.set_xlabel("n (samples)")
    ax2.set_ylabel("amplitude")

    for i, x in enumerate(
        [random.randrange(0, len(self.channels)) for _ in range(10)]
    ):
        ax2.plot(
            numpy.arange(self.frame_size),
            self.ytc[0][x][: self.frame_size],
            color="C{0}".format(i),
            linestyle="--",
            alpha=0.5,
            label="{0}Hz".format(round(self.channels[x], 2)),
        )
    i += 1

    ax2.grid()
    ax2.legend(loc="upper right")

    plt.show()


def _auditory_filterbank(x, fc, fs):
    J = 4

    # bc3db = -3/J dB
    A = numpy.exp(-(3 / J) * numpy.pi / (fs * numpy.sqrt(2 ** (1 / J) - 1)))

    cos_theta1 = (1 + A * A) / (2 * A) * numpy.cos(2 * numpy.pi * fc / fs)
    cos_theta2 = (2 * A) / (1 + A * A) * numpy.cos(2 * numpy.pi * fc / fs)
    rho1 = (1 / 2) * (1 - A * A)
    rho2 = (1 - A * A) * numpy.sqrt(1 - cos_theta2 ** 2)

    resonator_1_b = [rho1, 0, -rho1]
    resonator_1_a = [1, -A * cos_theta1, A * A]

    resonator_2_b = [rho2]
    resonator_2_a = [1, -A * cos_theta2, A * A]

    x = scipy.signal.lfilter(resonator_1_b, resonator_1_a, x)
    x = scipy.signal.lfilter(resonator_1_b, resonator_1_a, x)
    x = scipy.signal.lfilter(resonator_2_b, resonator_2_a, x)
    x = scipy.signal.lfilter(resonator_2_b, resonator_2_a, x)

    return x


"""
i would've used the sacf() function here from method 1
but the IFFT is specific to that method and we're using weighted salience for periodicity analysis here
"""


def _bandwise_summary_spectrum(
    x_channels: typing.List[numpy.ndarray], k=None
) -> numpy.ndarray:
    # k is same as p (power) in the Klapuri/Ansi paper
    if not k:
        k = 0.67

    shape = int((x_channels[0].shape[0]-1)/2)
    running_sum = numpy.zeros(shape)

    for xc in x_channels:
        running_sum += numpy.abs(numpy.fft.fft(xc)[:shape]) ** k

    return running_sum


def _weight(tau_low, tau_up, fs, epsilon1, epsilon2, m):
    return (fs / tau_low + epsilon1) / (m * fs / tau_up + epsilon2)


@njit
def _max_Ut(Ut, tau, delta_tau, m, K):
    Ut_max = 0.0
    for tau_adjusted in numpy.linspace(tau + delta_tau / 2, tau - delta_tau / 2):
        if tau_adjusted == 0.0:
            continue
        k_tau_m = int(round(m * K / tau_adjusted))
        if 0 <= k_tau_m < Ut.shape[0]:
            Ut_max = max(Ut[k_tau_m], Ut_max)
    return Ut_max


def smax(q: float, Ur: numpy.ndarray) -> float:
    tor = 0.5*0.2
    return tor
''' 
float PeriodicityAnalysis::smax(int q, float * Ur) {
    
    float tor = 0.5f*(torlow_[q] + torup_[q]);
    float deltator = torup_[q] - torlow_[q];
    
    //2745.4833984375 = 10.7666015625*255 = (44100.0/4096) * 255
    
    
    int topm = tor*2745.4833984375; //2745.4833984375/(1.0/tor)
    
    if(topm>20) topm = 20; 
    
    //break early if go outside first 256 bins (later may allow use of wider band mainFFT for higher k ignoring band wise channel data? 
    
    float salience = 0.0f; 
    float srovertor = g_samplingrate/torup_[q];  //g_samplingrate/tor; 
    
    //float K = 4096.0/g_samplingrate; 
    
    for (int m=1; m<topm; ++m) {
        
        int lowk = m*K_/(tor+0.5*deltator) +0.5f; //dd 0.5 since rounding to nearest integer
        int highk = m*K_/(tor-0.5*deltator) + 0.5f;
        
        //indexing safety check
        if((lowk < UKSIZE) && (highk < UKSIZE)) { 
            
            float maxu = Ur[lowk];
            
            for(int i=lowk+1; i<highk; ++i) {
                
                float nowu = Ur[i];
                
                if(nowu>maxu)
                    maxu = nowu; 
                
            }
            
            float w = 1.0/(m*srovertor + 320.0f);
            
            salience += w* maxu; 
            
        }
        
    }
    
    //was srovertor
    salience *= g_samplingrate/torlow_[q] + 5.f; //numerator multiplier from w doesn't depend on m and taken outside 
    
    //smax_[q] = salience; 
    
    return salience; 
    
}
'''
