from chord_detection.esacf import MultipitchESACF
from chord_detection.harmonic_energy import MultipitchHarmonicEnergy
from chord_detection.iterative_f0 import MultipitchIterativeF0
from chord_detection.prime_multif0 import MultipitchPrimeMultiF0
from chord_detection.multipitch import METHODS

import sys
import argparse


def main_cli():
    method_nums = [k for k in METHODS.keys()]
    method_nums_help_string = "-1 = all, "
    for k in METHODS.keys():
        method_nums_help_string += "{0} ({1}), ".format(k, METHODS[k].display_name())

    method_nums_help_string = method_nums_help_string[:-2]  # strip trailing ', '

    parser = argparse.ArgumentParser(
        prog="chord-detection",
        description="Collection of chord-detection techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--key",
        action="store_true",
        help="estimate the key using the Krumhansl-Schmuckler key-finding algorithm",
    )
    parser.add_argument(
        "--displayplots",
        type=int,
        help="display intermediate plots at specified frame with matplotlib",
        default=-1,
    )
    parser.add_argument(
        "--method",
        type=int,
        help=method_nums_help_string,
        default=next(iter(METHODS.keys())),
    )
    parser.add_argument("input_path", help="Path to WAV audio clip")
    args = parser.parse_args()

    compute_objs = []

    if args.method == -1:
        for v in METHODS.values():
            compute_objs.append(v(args.input_path))
    else:
        try:
            compute_objs.append(METHODS[args.method](args.input_path))
        except KeyError:
            raise ValueError("valid methods: {0}".format(method_nums_help_string))

    for compute_obj in compute_objs:
        print(
            "{0} - {1}".format(compute_obj.method_number(), compute_obj.display_name())
        )
        chromagram = compute_obj.compute_pitches(args.displayplots)
        print(chromagram)
        if args.key:
            print(chromagram.key())


if __name__ == "__main__":
    main_cli()
