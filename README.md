This repository is an MIT-licensed collection of multipitch/polyphonic instrument chord and key detection methods, implemented from academic papers using Python.

A chromagram and PCP (pitch class profile) are the same thing - an array of the 12 notes of Western music to describe a chord with some value (energy, etc.).

### Docker instructions

Build the Docker container:
```
sevagh:chord-detection $ docker build -qt chord-detection .
sha256:461b836cc2604d1309ee14d91735de1a16474cf5488a5bf3b24f3f83851f9c54
```

Run the Docker container with a WAV file containing a chord you want analyzed.

Example with ESACF and a piano C chord (from freesound.org):

```
sevagh:chord-detection $ docker run chord-detection 68441__pinkyfinger__piano-c.wav
Using method 1 - ESACF+chromagram
{   'A': 0,
    'A#': 0.792808013506129,
    'B': 0,
    'C': 1.6671218971286224,
    'C#': 0,
    'D': 1.0,
    'D#': 0.2586313270705468,
    'E': 0,
    'F': 1.2124568286131734,
    'F#': 0,
    'G': 0.6419538129478229,
    'G#': 1.1947754844241687}
```

Example with ESACF and an acoustic guitar G major 7 chord (from freesound.org):

```
sevagh:chord-detection $ docker run chord-detection 439554__inspectorj__guitar-acoustic-gmaj7-chord.wav
Using method 1 - ESACF+chromagram
{   'A': 0.18735402700055612,
    'A#': 0.752525570293385,
    'B': 0,
    'C': 1.0191414372440775,
    'C#': 0.19625453908915255,
    'D': 0.7832300698419827,
    'D#': 0,
    'E': 0,
    'F': 0.6266419782791928,
    'F#': 0.28873167023959895,
    'G': 1.366173867147784,
    'G#': 0.31705291768459837}
```

### Methods

Use the list numbering (e.g. ESACF = 1) as values for the `--method` argument.

1. ESACF (enhanced summary autocorrelation, [1], [2])

_[1] T. Tolonen and M. Karjalainen, "A computationally efficient multipitch analysis model," in IEEE Transactions on Speech and Audio Processing, vol. 8, no. 6, pp. 708-716, Nov. 2000._

_[2] V. Zenz and A. Rauber, "Automatic Chord Detection Incorporating Beat and Key Detection," 2007 IEEE International Conference on Signal Processing and Communications, Dubai, 2007, pp. 1175-1178._
