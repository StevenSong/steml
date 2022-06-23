import os
import logging
import pandas as pd
from typing import Dict, List
from multiprocessing import Process
from steml.defines import SIZE
from steml.utils import config_logger, LogLevel


def train(
    label: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    activation: str,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    lr_reduction: float,
    lr_patience: int,
    loss: str,
    metrics: List[str],
    output_dir: str,
    num_workers: int,
    gpu_config: Dict[int, int],
    log_level: LogLevel = LogLevel.INFO,
    skip_log_config: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if not skip_log_config:
        log_file = os.path.join(output_dir, 'train.log')
        config_logger(log_level=log_level, log_file=log_file)

    from steml.utils import config_gpus
    config_gpus({gpu: mem for gpu, mem in gpu_config.items()})

    from steml.data.dataset import make_dataset
    from steml.models import make_resnet18, get_callbacks
    from tensorflow.keras.models import load_model

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    num_classes = 2

    train_ds = make_dataset(paths=train_df['path'], labels=train_df[label], num_classes=num_classes, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_ds = make_dataset(paths=val_df['path'], labels=val_df[label], num_classes=num_classes, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model = make_resnet18(
        input_shape=(SIZE, SIZE, 3),
        num_classes=num_classes,
        activation=activation,
        lr=lr,
        loss=loss,
        metrics=metrics,
    )

    model_file = os.path.join(output_dir, 'model.h5')
    history = model.fit(
        x=train_ds,
        epochs=epochs,
        verbose=1,
        validation_data=val_ds,
        callbacks=get_callbacks(
            model_file=model_file,
            patience=patience,
            lr_reduction=lr_reduction,
            lr_patience=lr_patience,
        ),
    )
    history = pd.DataFrame(history.history)
    history.index.name = 'epoch'
    history.to_csv(os.path.join(output_dir, 'training_history.csv'))

    model = load_model(model_file)

    test_ds = make_dataset(paths=test_df['path'], labels=test_df[label], num_classes=num_classes, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    y_hat_test = model.predict(test_ds)[:, 1]
    pd.DataFrame({'path': test_df['path'], label: y_hat_test}).to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)


def train_random_splits(
    num_splits: int,
    label: str,
    input_dir: str,
    output_dir: str,
    activation: str,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    lr_reduction: float,
    lr_patience: int,
    loss: str,
    metrics: List[str],
    num_workers: int,
    gpu_config: Dict[int, int],
    log_level: LogLevel = LogLevel.INFO,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'nested_cross_validate.log')
    config_logger(log_level=log_level, log_file=log_file)

    # setup trials
    from steml.data import get_tile_paths_labels, get_random_splits
    paths, labels = get_tile_paths_labels(input_dir=input_dir, label=label)
    splits = get_random_splits(num_splits=num_splits, paths=paths, labels=labels)
    for trial, (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    ) in enumerate(splits):
        trial_dir = os.path.join(output_dir, str(trial))
        os.makedirs(trial_dir, exist_ok=True)
        train_csv = os.path.join(trial_dir, 'train.csv')
        val_csv = os.path.join(trial_dir, 'val.csv')
        test_csv = os.path.join(trial_dir, 'test.csv')
        pd.DataFrame({'path': train_paths, label: train_labels}).to_csv(train_csv, index=False)
        pd.DataFrame({'path': val_paths, label: val_labels}).to_csv(val_csv, index=False)
        pd.DataFrame({'path': test_paths, label: test_labels}).to_csv(test_csv, index=False)

    # run all trials
    for trial in range(num_splits):
        trial_dir = os.path.join(output_dir, str(trial))
        train_csv = os.path.join(trial_dir, 'train.csv')
        val_csv = os.path.join(trial_dir, 'val.csv')
        test_csv = os.path.join(trial_dir, 'test.csv')
        logging.info(f'Running Trial {trial}')
        p = Process(
            target=train,
            name=f'Trial {trial}',
            kwargs={
                'label': label,
                'train_csv': train_csv,
                'val_csv': val_csv,
                'test_csv': test_csv,
                'activation': activation,
                'batch_size': batch_size,
                'epochs': epochs,
                'patience': patience,
                'lr': lr,
                'lr_reduction': lr_reduction,
                'lr_patience': lr_patience,
                'loss': loss,
                'metrics': metrics,
                'output_dir': trial_dir,
                'num_workers': num_workers,
                'gpu_config': gpu_config,
                'skip_log_config': True,
            },
        )
        p.start()
        p.join()
