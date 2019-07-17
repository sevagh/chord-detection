from chord_detection import MultipitchESACF 
import sys


if __name__ == '__main__':
    esacf = MultipitchESACF(sys.argv[1], n_peaks=2)
    print(esacf.compute_pitches())
    esacf.display_plots()
