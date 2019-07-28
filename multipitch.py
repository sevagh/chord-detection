from chord_detection import (
    MultipitchESACF,
    MultipitchHarmonicEnergy,
    MultipitchIterativeF0,
    MultipitchPrimeMultiF0,
)
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="chord-detection",
        description="Collection of chord-detection techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--key", help="estimate the key of input audio clip")
    parser.add_argument(
        "--bitstring", action="store_true", help="emit a 12-note chromagram bitstring"
    )
    parser.add_argument(
        "--displayplots",
        action="store_true",
        help="display intermediate plots with matplotlib",
    )
    parser.add_argument(
        "--method",
        type=int,
        help="-1 = all, 1 = ESACF, 2 = Harmonic Energy, 3 = Iterative F0, 4 = Prime-multiF0",
        default=1,
    )
    parser.add_argument("input_path", help="Path to WAV audio clip")
    args = parser.parse_args()

    compute_objs = []

    if args.method == 1 or args.method == -1:
        compute_objs.append(MultipitchESACF(args.input_path))
    if args.method == 2 or args.method == -1:
        compute_objs.append(MultipitchHarmonicEnergy(args.input_path))
    if args.method == 3 or args.method == -1:
        compute_objs.append(MultipitchIterativeF0(args.input_path))
    if args.method == 4 or args.method == -1:
        compute_objs.append(MultipitchPrimeMultiF0(args.input_path))

    if not compute_objs:
        raise ValueError("valid methods: -1 (all), 1, 2, 3, 4")

    for compute_obj in compute_objs:
        print(compute_obj.display_name())

        chromagram = compute_obj.compute_pitches()
        if args.bitstring:
            print(chromagram.pack())
        else:
            print(chromagram)

        if args.displayplots:
            compute_obj.display_plots()
