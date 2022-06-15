import argparse


def slice_parser(subparsers):
    slice = subparsers.add_parser(
        'slice',
        help='Slice brightfield HE image into tiles.',
    )
    slice.add_argument(
        '--image',
        required=True,
        help='Path to brightfield HE image.',
    )
    slice.add_argument(
        '--scaling_factors',
        required=True,
        help='Path to scale factor JSON from spaceranger count.',
    )
    slice.add_argument(
        '--tissue_positions',
        required=True,
        help='Path to tissue positions CSV from spaceranger count.',
    )
    slice.add_argument(
        '--output',
        required=True,
        help='Path to output folder.',
    )
    return slice


def label_parser(subparsers):
    label = subparsers.add_parser(
        'label',
        help='Generate binary label per barcode based '
             'on gene expression profiling.',
    )
    label.add_argument(
        '--feature_barcode_matrix',
        required=True,
        help='Path to folder containing feature-barcode '
             'matrix from spaceranger count.',
    )
    label.add_argument(
        '--conditions',
        required=True,
        help='Path to gene expression labeling conditions CSV, '
             'see implementation for more details.',
    )
    label.add_argument(
        '--name',
        required=True,
        help='Name of binary label.',
    )
    label.add_argument(
        '--output',
        required=True,
        help='Path to output folder.',
    )
    return label


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='recipe')
    subparsers.required = True

    slice = slice_parser(subparsers)
    label = label_parser(subparsers)

    args = parser.parse_args()
    return args
