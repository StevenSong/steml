#!/usr/bin/env python

from steml.recipes import train_nested_leave_one_out

train_nested_leave_one_out(
    label='lymphocyte',
    num_classes=2,
    input_dir='/mnt/data5/output/tiles/gi-infection-scaled',
    cache=True,
    shuffle_train=True,
    augment_train=True,
    balance_train=True,
    shuffle_val=True,
    augment_val=True,
    balance_val=True,
    output_dir=f'/mnt/data5/output/train/nested-leave-one-out-balanced',
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
    multi_gpu_config=[(0, -1)],
)
