from chord_detection import MultipitchESACF, MultipitchHarmonicEnergy
import sys
import pprint
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="chord-detection",
        description="Collection of chord-detection techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-k", "--key", help="estimate the key of input audio clip")
    parser.add_argument(
        "--method", type=int, help="choose the method (see the README)", default=1
    )
    parser.add_argument(
        "--display-plots",
        action="store_true",
        help="display matplotlib plots for method",
    )
    parser.add_argument("input_path", help="Path to WAV audio clip")
    args = parser.parse_args()

    compute_obj = None

    if args.method == 1:
        print("Using method 1 - ESACF+chromagram")
        compute_obj = MultipitchESACF(args.input_path)
    elif args.method == 2:
        print("Using method 2 - harmonic energy")
        compute_obj = MultipitchHarmonicEnergy(args.input_path)
    else:
        raise ValueError("valid methods: 1")

    chromagram = compute_obj.compute_pitches()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(chromagram)

    if args.display_plots:
        compute_obj.display_plots()
