from chord_detection import MultipitchESACF
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

    esacf = MultipitchESACF(args.input_path)

    if args.method == 1:
        print("Using method 1 - ESACF+chromagram")

        chromagram = esacf.compute_pitches()
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(chromagram)

        if args.display_plots:
            esacf.display_plots()
    else:
        raise ValueError("valid methods: 1")
