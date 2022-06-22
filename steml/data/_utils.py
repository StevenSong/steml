import os
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import StratifiedKFold


Paths = List[str]
Labels = List[int]
Paths_Labels = Tuple[Paths, Labels]


def get_tile_paths_labels(input_dir: str, label: str) -> Paths_Labels:
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
    logging.info(f'Found {n} tiles')
    logging.info(f'{label}: {pos}/{n} ({pos/n:0.3f})')
    return list(paths), list(labels)


def nested_k_fold_split(
    k: int,
    paths: Paths,
    labels: Labels,
    seed: int = 2022,
) -> List[Tuple[List[Tuple[Paths_Labels, Paths_Labels]], Paths_Labels]]:
    paths = np.array(paths)
    labels = np.array(labels)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    ret = []
    for inner_idx, test_idx in skf.split(paths, labels):
        test_paths = paths[test_idx].tolist()
        test_labels = labels[test_idx].tolist()
        inner_paths = paths[inner_idx]
        inner_labels = labels[inner_idx]
        inner = []
        for train_idx, val_idx in skf.split(inner_paths, inner_labels):
            train_paths = inner_paths[train_idx].tolist()
            train_labels = inner_labels[train_idx].tolist()
            val_paths = inner_paths[val_idx].tolist()
            val_labels = inner_labels[val_idx].tolist()
            inner.append((
                (train_paths, train_labels),
                (val_paths, val_labels),
            ))
        ret.append((inner, (test_paths, test_labels)))
    return ret
