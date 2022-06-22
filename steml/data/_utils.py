import os
import pandas as pd
from typing import List, Tuple


def get_tile_paths_labels(input_dir: str, label: str) -> Tuple[List[str], List[int]]:
    paths = []
    labels = []
    for path, dirs, files in os.walk(input_dir):
        for file in files:
            if file == f'{label}.csv':
                df = pd.read_csv(os.path.join(path, file))
                paths.append(path + '/' + df['barcode'] + '.png')
                labels.append(df[label])
    paths = pd.concat(paths)
    labels = pd.concat(labels)
    n = len(paths)
    pos = labels.sum()
    print(f'Found {n} tiles')
    print(f'{label}: {pos}/{n} ({pos/n:0.3f})')
    return list(paths), list(labels)
