#!/usr/bin/env python

from steml.recipes import train_leave_one_out

train_leave_one_out(
    label='lymphocyte',
    num_classes=2,
    num_inner_splits=5,
    input_dir='/mnt/data5/output/tiles/gi-infection-scaled',
    cache=True,
    augment_train=True,
    output_dir=f'/mnt/data5/output/train/leave-one-out-augment',
    activation='softmax',
    batch_size=256,
    epochs=100,
    lr=0.001,
    # lr_reduction=0.1,
    # num_reductions=5,
    lr_reduction=None,
    num_reductions=None,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
    num_workers=4,
    gpu_config={0: -1},
)
