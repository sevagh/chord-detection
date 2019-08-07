from ..chromagram import Chromagram
import numpy
import math
import librosa


_HAMMINGWINDOWNORM = [0.0011244659258033, 0.11559343551383, 0.42817348241183, 0.81822361914331, 1.0, 0.81822361914331, 0.42817348241183, 0.11559343551383, 0.0011244659258033]


'''
thank god for https://github.com/BansMarbol/PolyPitch
'''

class IterativeF0PeriodicityAnalysis():
    def __init__(
            self,
            fs: float,
            window_size: int,
            max_voices=4,
            tau_min=1.0/2100.0,
            tau_max=1.0/65.0,
            tau_prec=0.0000001,
            max_q=50,
        ):
        self.fs = fs
        self.window_size = window_size
        self.K = window_size/fs
        self.max_voices = max_voices
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_prec = tau_prec
        self.max_q = max_q

        self.voicesaliences = numpy.zeros(self.max_voices)
        self.voiceperiods = numpy.zeros(self.max_voices)
        self.smax = numpy.zeros(self.max_q)
        self.tau_low = numpy.zeros(self.max_q)
        self.tau_up = numpy.zeros(self.max_q)

    def compute(self, Uk: numpy.ndarray) -> Chromagram:
        num_voices_detected = 0
        cancellation_weight = 1.0
        polyphonyestimategamma = 0.66

        Ud = numpy.zeros(Uk.shape[0])
        Ur = numpy.array(Uk)

        # clear the arrays from the last run
        self.voicesaliences[:] = 0.0
        self.voiceperiods[:] = 0.0

        prevmixturescore = 0.0
        mixturescore = 0.0

        keepgoing = True

        while keepgoing:
            winningtau, bestsalience = self.min_search(Ur)
            self.voicesaliences[num_voices_detected] = bestsalience
            self.voiceperiods[num_voices_detected] = winningtau

            num_voices_detected += 1
            mixturescore += bestsalience

            testquantity = mixturescore/(math.pow(num_voices_detected, polyphonyestimategamma))

            if num_voices_detected >= self.max_voices or testquantity <= prevmixturescore:
                keepgoing = False
            else:
                prevmixturescore = testquantity
                tau = winningtau
                topm = int(tau*(self.fs/self.window_size)*Uk.shape[0])

                srovertau = self.fs/tau
                weight = srovertau + 5
                for m in range(1, topm):
                    partialK = m*self.K/tau + 0.5
                    if partialK <= Uk.shape[0]:
                        Urweight = Ur[int(partialK)]
                        Urweight *= weight/(m*srovertau + 320.0)

                        lowk = max(int(partialK-4), 0)
                        highk = min(int(partialK+4), 255)

                        for j in range(lowk, highk+1):
                            hammingindexnow = int(j - partialK + 4)
                            val = _HAMMINGWINDOWNORM[hammingindexnow] * Urweight
                            Ud[j] += val

                for i in range(Uk.shape[0]):
                    diff = Uk[i] - cancellation_weight*Ud[i]
                    Ur[i] = max(diff, 0)

        if num_voices_detected > 0:
            num_voices_detected -= 1

        c = Chromagram()
        for i in range(self.voiceperiods.shape[0]):
            note = librosa.hz_to_note(1.0/self.voiceperiods[i], octave=False)
            c[note] += self.voicesaliences[i]

        return c

    def min_search(self, Ur: numpy.ndarray) -> float:
        q = 0

        self.tau_low[0] = self.tau_min
        self.tau_up[0] = self.tau_max

        qbest = 0

        while (self.tau_up[qbest] - self.tau_low[qbest]) > self.tau_prec and q < self.max_q-1:
            q += 1
            self.tau_low[q] = (self.tau_low[qbest] + self.tau_up[qbest])*0.5
            self.tau_up[q] = self.tau_up[qbest]
            self.tau_up[qbest] = self.tau_low[q]

            self.smax[q] = self.smax_fn(q, Ur)
            self.smax[qbest] = self.smax_fn(qbest, Ur)

            whichq = 0
            maxval = self.smax[0]

            for j in range(1, q+1):
                valnow = self.smax[j]
                if valnow > maxval:
                    maxval = valnow
                    whichq = j
            qbest = whichq

        winningtau = (self.tau_low[qbest] + self.tau_up[qbest])*0.5
        return winningtau, self.smax[qbest]

    def smax_fn(self, q: int, Ur: numpy.ndarray) -> float:
        tau = 0.5*(self.tau_low[q] + self.tau_up[q])
        deltator = self.tau_up[q] - self.tau_low[q]

        topm = int(tau*(self.fs/self.window_size)*Ur.shape[0])
        if topm > 20:
            topm = 20

        salience = 0.0
        srovertau = self.fs/self.tau_up[q]

        for m in range(1, topm):
            lowk = int(m*self.K*(tau+0.5*deltator) + 0.5)
            highk = int(m*self.K/(tau-0.5*deltator) + 0.5)

            if lowk < Ur.shape[0] and highk < Ur.shape[0]:
                maxu = Ur[lowk]
                for i in range(lowk+1, highk):
                    maxu = max(Ur[i], maxu)
                w = 1.0/(m*srovertau + 320.0)
                salience += w*maxu

        salience *= self.fs/self.tau_low[q] + 5
        return salience
