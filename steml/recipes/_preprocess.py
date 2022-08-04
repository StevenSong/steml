import os
import gzip
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy.io import mmread, mmwrite
from typing import List, Tuple, Optional
from steml.defines import SIZE, LogLevel
from steml.utils import config_logger


Image.MAX_IMAGE_PIXELS = 1000000000


def lognorm(
    feature_barcode_matrix: str,
    output_dir: str,
    normalize: bool = True,
    target: int = 10000,
    use_slide_size: bool = True,
    log_scale: bool = True,
    log_level: LogLevel = LogLevel.INFO,
):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'lognorm.log')
    config_logger(log_level=log_level, log_file=log_file)

    if not normalize and not log_scale:
        raise ValueError('Nothing to do if both normalize == False and log_scale == False')
    if feature_barcode_matrix == output_dir:
        raise ValueError('Input and output directories are the same, would overwrite, aborting! Please make a copy')
    mat = mmread(os.path.join(feature_barcode_matrix, 'matrix.mtx.gz'))
    if normalize:
        t = mat.shape[1]  # number of spots in slide
        z = mat.sum()  # sum of all counts in slide
        t = t if use_slide_size else 1
        mat = mat * t / z * target  # (z/t) is average count per spot for the slide
    if log_scale:
        mat = mat.log1p()
    with gzip.open(os.path.join(output_dir, 'matrix.mtx.gz'), 'wb') as gz:
        mmwrite(gz, mat)


def slice(
    image: str,
    tissue_positions: str,
    output_dir: str,
    size: int = SIZE,
    resize: Optional[int] = None,
    resize_method: str = 'scale',
    log_level: LogLevel = LogLevel.INFO,
) -> None:
    """
    Slice a brightfield HE image into tiles.
    Tile locations are based off orange crate packing used in Visium slides.
    Visium slide spots are circular but we simplify by creating square tiles
    as the spots do not overlap. More on the packing method is described here:
    https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/images

    Parameters
    ----------
    image: <string> Path to brightfield HE image.
    tissue_positions: <string> Path to tissue positions CSV from spaceranger count.
    size: <int> Pixel length of square output tile.
    resize: <int> Optional new pixel length to resize tile to.
    resize_method: <string> Either 'scale' or 'pad'. Scale using
    output_dir: <string> Path to output folder.

    Returns
    -------
    Output tiles in PNG format named <row>_<col>_<barcode>.png in output folder.
    """

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'slice.log')
    config_logger(log_level=log_level, log_file=log_file)

    tps = pd.read_csv(
        tissue_positions,
        header=None,
        names=['barcode', 'inside', 'row', 'col', 'y', 'x'],
    )
    tps = tps[tps['inside'].astype(bool)].sort_values(by=['row', 'col'])
    tps = [r for _, r in tps[['barcode', 'x', 'y']].iterrows()]

    offset = size / 2
    with Image.open(image) as im:
        for barcode, x, y in tqdm(tps):
            left = int(x - offset)
            right = int(x + offset)
            top = int(y - offset)
            bottom = int(y + offset)
            tile = im.crop((left, top, right, bottom))

            if resize is not None and size != resize:
                if resize_method == 'scale':
                    tile = tile.resize((resize, resize), Image.ANTIALIAS)
                elif resize_method == 'pad':
                    if resize < size:
                        raise ValueError(f'Cannot resize tile smaller with padding (size {size} > {resize})')
                    shape = (resize, resize, len(tile.getbands()))
                    new_tile = Image.fromarray(np.zeros(shape, dtype=np.uint8))
                    corner = (resize - size) // 2
                    new_tile.paste(tile, (corner, corner))
                    tile = new_tile
                else:
                    raise ValueError(f'Unknown resize method: {resize_method}')

            tile.save(os.path.join(output_dir, f'{barcode}.png'))


def label(
    feature_barcode_matrix: str,
    conditions: List[List[Tuple[str, bool, int]]],
    name: Optional[str],
    tile_dir: str,
    output_dir: str,
    log_level: LogLevel = LogLevel.INFO,
) -> None:
    """
    Generate labels based on gene expression data.

    Parameters
    ----------
    feature_barcode_matrix: <string> Path to folder containing feature barcode
                            matrix from spaceranger count.
    conditions: <list of lists of tuples> A datastructure defining the condition
                for positively labeling a spot. The condition is interpreted in
                disjunctive normal form (DNF) with the following format.

                The expression level of a gene is defined by a 3-tuple of
                (gene, lte, threshold), where gene is a string of the gene name,
                lte is a boolean indicating if the observed expression level
                should be less than or equal to the given threshold, and threshold
                is an integer value for the target expression level. For example,
                to specify an expression level of BRCA1 > 2, use the 3-tuple:
                ('BRCA1', False, 2).

                The conditions given by a list of expression level tuples
                is then ANDed together, i.e. each list of tuples is a clause in
                disjunctive normal form. For example, to specify BRCA1 > 2 and
                BRCA2 <= 3, use the list of tuples:
                [('BRCA1', False, 2), ('BRCA2', True, 3)]

                Finally, a list of clauses are ORed together. For example, if the
                label was any spot expression BRCA1 > 2 or BRCA2 > 2, use the
                following list of list of tuples:
                [[('BRCA1', False, 2)], [('BRCA2', False, 2)]]
    name: <string> The name of the label and output CSV file.
    output_dir: <string> Path to output folder.

    Returns
    -------
    Saves a CSV in the output folder of calculated binary label per spot barcode.
    Also includes the gene expression data used to derive the binary label.
    """

    # https://en.wikipedia.org/wiki/Disjunctive_normal_form

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'label.log')
    config_logger(log_level=log_level, log_file=log_file)

    genes = {gene for c in conditions for gene, _, _ in c}

    features = pd.read_csv(
        f'{feature_barcode_matrix}/features.tsv.gz',
        sep='\t', header=None, names=['id', 'gene', 'type'],
    )
    all_genes = features['gene']
    barcodes = pd.read_csv(
        f'{feature_barcode_matrix}/barcodes.tsv.gz',
        sep='\t', header=None, usecols=[0], names=['barcode'],
    )['barcode']
    matrix = mmread(f'{feature_barcode_matrix}/matrix.mtx.gz')

    # transform to pandas DataFrame with rows as barcodes and cols as genes
    df = pd.DataFrame.sparse.from_spmatrix(matrix, index=all_genes, columns=barcodes).T

    # filter by target genes
    df = df[all_genes[(features['type'] == 'Gene Expression') & all_genes.isin(genes)]]
    df = df.sparse.to_dense()

    # parse through conjunctive clauses
    y = np.zeros(len(df)).astype(bool)
    clause_strs = []
    for clause in conditions:
        cy = np.ones(len(df)).astype(bool)
        gene_strs = []
        for gene, lte, threshold in clause:
            cy &= df[gene] <= threshold if lte else df[gene] > threshold
            gene_strs.append(f'{gene}{"≤" if lte else ">"}{threshold}')
        y |= cy
        clause_strs.append('(' + ' ∧ '.join(gene_strs) + ')')
    df[name] = y.astype(int)
    dnf_str = ' ∨ '.join(clause_strs)
    logging.info(f'labeled {df[name].sum()}/{len(df)} ({df[name].sum()/len(df):0.3f}) as {name} at {output_dir}')
    logging.info(dnf_str)
    df['path'] = tile_dir + '/' + df.index + '.png'
    df.to_csv(f'{output_dir}/{name}.csv')
