import os
import argparse


def slice_parser(subparsers):
    slice = subparsers.add_parser('slice', help='Slice brightfield HE image into tiles')
    slice.add_argument('--image', required=True, help='Path to brightfield HE image')
    slice.add_argument('--scaling_factors', required=True, help='Path to scale factor JSON from spaceranger count')
    slice.add_argument('--tissue_positions', required=True, help='Path to tissue positions CSV from spaceranger count')
    slice.add_argument('--output', required=True, help='Path to output folder')
    return slice


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='recipe')
    subparsers.required = True

    slice = slice_parser(subparsers)

    args = parser.parse_args()
    return args
