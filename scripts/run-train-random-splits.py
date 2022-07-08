#!/usr/bin/env python

from steml.recipes import train_random_splits

label = 'lymphocyte'

train_random_splits(
    num_splits=25,
    label=label,
    num_classes=2,
    input_dir='/mnt/data5/output/tiles/gi-infection-padded',
    output_dir=f'/mnt/data5/output/train/{label}-padded-tiles',
    activation='softmax',
    batch_size=256,
    epochs=1000,
    patience=15,
    lr=0.01,
    lr_reduction=0.1,
    lr_patience=5,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
    num_workers=4,
    gpu_config={0: -1},
)
