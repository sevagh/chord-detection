from esacf import multipitch_esacf
import matplotlib.pyplot as plt
import numpy
import sys


if __name__ == '__main__':
    x, sacf, esacf, interesting = multipitch_esacf(sys.argv[1])
    samples = numpy.arange(len(x))

    fig1, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title('x[n] (piano chord)')
    ax1.set_xlabel('n (samples)')
    ax1.set_ylabel('amplitude')
    ax1.plot(samples, x, 'b', alpha=0.8, label='x[n]')
    ax1.grid()
    ax1.legend(loc='upper right')

    ax2.set_title('SACF, ESACF')
    ax2.set_xlabel('n (samples)')
    ax2.set_ylabel('amplitude')
    ax2.plot(samples, sacf, 'g', linestyle='--', alpha=0.5, label='sacf')
    ax2.plot(samples, esacf, 'b', linestyle=':', alpha=0.5, label='esacf')

    ax2.grid()
    ax2.legend(loc='upper right')

    plt.axis('tight')
    fig1.tight_layout()
    plt.show()
