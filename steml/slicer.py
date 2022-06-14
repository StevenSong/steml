import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000


def slice(
    image: str,
    scaling_factors: str,
    tissue_positions: str,
    output: str,
) -> None:
    """
    slice a brightfield HE image into tiles
    args:
        image: path to brightfield HE image
        scaling_factors: path to scale factor JSON from spaceranger count
        tissue_positions: path to tissue positions CSV from spaceranger count
        output: path to output folder
    outputs:
        tiles in PNG format named <row>_<col>.png
    """
    # slice is based off orange crate packing method described here:
    # https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/images
    # simplify spots by expanding to square tiles which do not overlap

    os.makedirs(output)

    with open(scaling_factors) as f:
        sfs = json.load(f)
        offset = sfs['spot_diameter_fullres'] / 2

    tps = pd.read_csv(
        tissue_positions,
        header=None,
        names=['barcode', 'inside', 'row', 'col', 'y', 'x'],
    )
    tps = tps[tps['inside'].astype(bool)].sort_values(by=['row', 'col'])
    tps = [r for _, r in tps[['row', 'col', 'x', 'y']].iterrows()]

    with Image.open(image) as im:
        for row, col, x, y in tqdm(tps):
            left = int(x - offset)
            right = int(x + offset)
            top = int(y - offset)
            bottom = int(y + offset)
            tile = im.crop((left, top, right, bottom))
            tile.save(os.path.join(output, f'{row}_{col}.png'))

