#!/usr/bin/env python

from steml.recipes import train

input_dir = '/mnt/data5/output/tiles/gi-infection-scaled'
label = 'epithelium'

train(
    label=label,
    num_classes=2,
    train_csv='/mnt/data5/output/tiles/gi-infection-scaled/C.diff/C1/epithelium.csv',
    activation='softmax',
    batch_size=256,
    epochs=50,
    lr=0.01,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy', 'AUC'],
    output_dir='/mnt/data5/output/train/colon-epithelium',
    num_workers=4,
    callback_monitor='loss',
    cache=True,
    shuffle_train=True,
    augment_train=True,
    balance_train=False,
    test_csv='/mnt/data5/output/tiles/gi-infection-scaled/C.diff/B1/epithelium.csv',
    gpu_config=(0, -1),
)
