#!/usr/bin/env python

import os
import pandas as pd
from steml.recipes import train_loo

input_dir = '/mnt/data5/output/tiles/gi-infection-scaled'
label = 'epithelium'

samples = []
for slide in ['H.pylori', 'C.diff']:
    for section in ['A1', 'B1', 'C1', 'D1']:
        if slide == 'C.diff' and section == 'D1':
            continue
        root = os.path.join(input_dir, slide, section)
        df = pd.read_csv(os.path.join(root, f'{label}.csv'))
        df['path'] = root + '/' + df['barcode'] + '.png'
        samples.append((f'{slide}.{section}', df))

train_loo(
    label=label,
    num_classes=2,
    samples=samples,
    cache=True,
    shuffle_train=True,
    augment_train=True,
    balance_train=False,
    output_dir=f'/mnt/data5/output/train/colon-epithelium-loo',
    activation='softmax',
    batch_size=256,
    epochs=50,
    patience=100,  # has no effect
    lr=0.01,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy', 'AUC'],
    callback_monitor='loss',
    min_delta=0.1,
    num_workers=4,
    multi_gpu_config=[(0, -1), (1, -1)],
)
