This repository is an MIT-licensed collection of multipitch/polyphonic instrument chord and key detection methods, implemented from academic papers using Python.

A chromagram and PCP (pitch class profile) are the same thing - an array of the 12 notes of Western music to describe a chord with some value (energy, etc.). Note that the float values of the chromagram in each method most likely represent different measures and units.

### Docker instructions

Build the Docker container:
```
sevagh:chord-detection $ docker build -qt chord-detection .
sha256:461b836cc2604d1309ee14d91735de1a16474cf5488a5bf3b24f3f83851f9c54
```

Run the Docker container with a WAV file containing a chord you want analyzed.

Example with ESACF and a Gmaj7 acoustic guitar chord:

```
$ docker run chord-detection guitar-acoustic-gmaj7-chord.wav --method 1
Using method 1 - ESACF+chromagram
OrderedDict([('C', 0.746),
             ('C#', 0.144),
             ('D', 0.573),
             ('D#', 0.0),
             ('E', 0.0),
             ('F', 0.459),
             ('F#', 0.211),
             ('G', 1.0),
             ('G#', 0.232),
             ('A', 0.137),
             ('A#', 0.551),
             ('B', 0.0)])
```

### Methods

Use the list numbering (e.g. ESACF = 1) as values for the `--method` argument.

1. ESACF (enhanced summary autocorrelation, [1], [2])
2. Harmonic energy chromagram ([3])

### Results

To make the results readable, we used the `--bitstring` flag to round the normalized chromagram floats (0.0 <= cf <= 1.0) to a bit (0, 1). This is a very unscientific/quick'n'dirty test with random WAV files I had lying around. The instruments may be out of tune, my determination of "correctness" might be wrong. Also, chord detection is a hard problem.

Gmaj7 acoustic guitar:

|                  | C-C#-D-D#-E-F-F#-G-G#-A-A#-B |
| ---------------- | ------------ |
| Correct          | 001000110001 |
| ESACF            | 101000010010 |
| harmonic energy  | 001000100000 |

Fmajor3 piano:

|                  | C-C#-D-D#-E-F-F#-G-G#-A-A#-B |
| ---------------- | ------------ |
| Correct          | 100010000100 |
| ESACF            | 010001101100 |
| harmonic energy  | 101001110110 |

Dmajor acoustic guitar:

|                  | C-C#-D-D#-E-F-F#-G-G#-A-A#-B |
| ---------------- | ------------ |
| Correct          | 001000100100 |
| ESACF            | 010100101000 |
| harmonic energy  | 011000100100 |

C acoustic guitar:

|                  | C-C#-D-D#-E-F-F#-G-G#-A-A#-B |
| ---------------- | ------------ |
| Correct          | 100010010001 |
| ESACF            | 010000000100 |
| harmonic energy  | 101000011001 |

C piano:

|                  | C-C#-D-D#-E-F-F#-G-G#-A-A#-B |
| ---------------- | ------------ |
| Correct          | 100010010000 |
| ESACF            | 101001001000 |
| harmonic energy  | 100000000000 |

### References

_[1] T. Tolonen and M. Karjalainen, "A computationally efficient multipitch analysis model," in IEEE Transactions on Speech and Audio Processing, vol. 8, no. 6, pp. 708-716, Nov. 2000._

_[2] V. Zenz and A. Rauber, "Automatic Chord Detection Incorporating Beat and Key Detection," 2007 IEEE International Conference on Signal Processing and Communications, Dubai, 2007, pp. 1175-1178._

_[3] M Stark, Adam and Plumbley, Mark., "Real-Time Chord Recognition for Live Performance," in Proceedings of the 2009 International Computer Music Conference (ICMC 2009), Montreal, Canada, 16-21 August 2009._
