#!/usr/bin/env python

import os
import pandas as pd
from steml.recipes import train_outer_loo_inner_cv

input_dir = '/mnt/data5/output/tiles/gi-infection-scaled'
label = 'lymphocyte'

samples = []
for slide in ['H.pylori', 'C.diff']:
    for section in ['A1', 'B1', 'C1', 'D1']:
        if slide == 'C.diff' and section == 'D1':
            continue
        root = os.path.join(input_dir, slide, section)
        df = pd.read_csv(os.path.join(root, f'{label}.csv'))
        df['path'] = root + '/' + df['barcode'] + '.png'
        samples.append((f'{slide}.{section}', df))

train_outer_loo_inner_cv(
    label=label,
    num_classes=2,
    num_inner_splits=4,
    samples=samples,
    cache=True,
    shuffle_train=True,
    augment_train=True,
    balance_train=False,
    output_dir='/mnt/data5/output/train/leave-one-out-lr',
    activation='softmax',
    batch_size=256,
    epochs=100,
    lr=0.01,
    lr_reduction=0.1,
    num_reductions=5,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
    num_workers=4,
    multi_gpu_config=[(0, -1), (1, -1)],
)
