import pandas as pd
from steml.arguments import parse_args
from steml.preprocessors import slice, label


def run(args):
    if args.recipe == 'slice':
        slice(
            image=args.image,
            scaling_factors=args.scaling_factors,
            tissue_positions=args.tissue_positions,
            output=args.output,
        )
    elif args.recipe == 'label':
        # load gene expression label conditions into list of lists of 3-tuples
        # this is interpreted as a conditional in disjunctive normal form
        # see the implementation of `label` for more details
        conds = pd.read_csv(args.conditions)
        conditions = (
            conds.groupby('clause')
            .apply(
                lambda d: d[['gene', 'lte', 'threshold']].to_numpy().tolist()
            )
            .to_list()
        )
        label(
            feature_barcode_matrix=args.feature_barcode_matrix,
            conditions=conditions,
            name=args.name,
            output=args.output,
        )
    else:
        raise ValueError('No recipe specified')


if __name__ == '__main__':
    args = parse_args()
    run(args)
