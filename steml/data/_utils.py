import os
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split


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


def get_random_splits(
    num_splits: int,
    paths: Paths,
    labels: Labels,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> List[Tuple[Paths_Labels, Paths_Labels, Paths_Labels]]:
    ret = []
    n = len(paths)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    df = pd.DataFrame({'path': paths, 'labels': labels})
    for i in range(num_splits):
        df_train_val, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=i,
            shuffle=True,
            stratify=df['label'],
        )
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=val_size,
            random_state=i,
            shuffle=True,
            stratify=df_train_val['label'],
        )
        ret.append((
            (df_train['path'].tolist(), df_train['label'].tolist()),
            (df_val['path'].tolist(), df_val['label'].tolist()),
            (df_test['path'].tolist(), df_test['label'].tolist()),
        ))
    return ret
