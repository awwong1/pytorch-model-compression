#!/usr/bin/env python3
"""Helper script to analyze nvprof output."""
from argparse import ArgumentParser
from torch.autograd.profiler import load_nvprof


def main(path):
    trace = load_nvprof(path)
    print(trace)


if __name__ == "__main__":
    parser = ArgumentParser("nvprof analysis script")
    parser.add_argument("-p", "--path", type=str,
                        default="out.prof", help="path to out.prof file")
    args = parser.parse_args()
    main(args.path)
