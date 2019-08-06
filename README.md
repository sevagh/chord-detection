This repository is an MIT-licensed collection of multipitch/polyphonic instrument chord and key detection methods, implemented from academic papers using Python.

### Methods

Results of a piano C major chord. The outputs are the summed chromagrams across all the frames, normalized to 9 (the biggest single-digit number), in a 12-digit string of ints representing the chromatic scale, aka `C-C#-D-D#-E-F-F#-G-G#-A-A#-B`.

The expected notes of a C major chord are C E G, so the expected answer should resemble `100010010000`.

#### ESACF (Tolonen, Karjalainen)

```
reference: 100010010000
computed:  900003001000, key: Cmaj
```

![esacf](.github/piano_c_1.png)

_T. Tolonen and M. Karjalainen, "A computationally efficient multipitch analysis model," in IEEE Transactions on Speech and Audio Processing, vol. 8, no. 6, pp. 708-716, Nov. 2000._

_V. Zenz and A. Rauber, "Automatic Chord Detection Incorporating Beat and Key Detection," 2007 IEEE International Conference on Signal Processing and Communications, Dubai, 2007, pp. 1175-1178._

#### Harmonic Energy (Stark, Plumbley)

```
reference: 100010010000
computed:  921111111111, key: Cmin
```

![harmeng](.github/piano_c_2.png)

_M Stark, Adam and Plumbley, Mark., "Real-Time Chord Recognition for Live Performance," in Proceedings of the 2009 International Computer Music Conference (ICMC 2009), Montreal, Canada, 16-21 August 2009._

#### Iterative F0 (Klapuri, Anssi)

```
incomplete
```

![iterativef0](.github/piano_c_3.png)

_Klapuri, Anssi, "Multipitch Analysis of Polyphonic Music and Speech Signals Using an Auditory Model," IEEE TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 16, NO. 2, FEBRUARY 2008 255._

_Klapuri, Anssi. "Multiple Fundamental Frequency Estimation by Summing Harmonic Amplitudes." ISMIR (2006)._

#### Prime-multiF0 (Camacho, Kaver-Oreamuno)

```
reference: 100010010000
computed:  951000000002, key: Cmin
```

![primemultif0](.github/piano_c_4.png)

_Camacho, A, Oreamuno, I, "A multipitch estimation algorithm based on fundamental frequencies and prime harmonics," Sound and Music Computing Conference 2013._
