from steml.arguments import parse_args
from steml.slicer import slice


def run(args):
    if args.recipe == 'slice':
        slice(
            image=args.image,
            scaling_factors=args.scaling_factors,
            tissue_positions=args.tissue_positions,
            output=args.output,
        )
    else:
        raise ValueError('No recipe specified')


if __name__ == '__main__':
    args = parse_args()
    run(args)
