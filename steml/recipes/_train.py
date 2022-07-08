import os
import logging
import numpy as np
import pandas as pd
from time import sleep
from collections import defaultdict
from multiprocessing import Process
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.model_selection import StratifiedKFold
from steml.defines import SIZE, GPU_CONFIG
from steml.utils import config_logger, LogLevel


def train(
    label: str,
    num_classes: int,
    train_csv: str,
    activation: str,
    batch_size: int,
    epochs: int,
    lr: float,
    loss: str,
    metrics: List[str],
    output_dir: str,
    num_workers: int,
    callback_monitor: str,
    cache: bool = False,
    shuffle_train: bool = True,
    augment_train: bool = True,
    balance_train: bool = False,
    val_csv: Optional[str] = None,
    shuffle_val: bool = True,
    augment_val: bool = True,
    balance_val: bool = False,
    test_csv: Optional[str] = None,
    patience: Optional[int] = None,
    lr_reduction: Optional[float] = None,
    lr_patience: Optional[int] = None,
    num_reductions: Optional[int] = None,
    gpu_config: Optional[Tuple[int, int]] = None,
    log_level: LogLevel = LogLevel.INFO,
    skip_log_config: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if not skip_log_config:
        log_file = os.path.join(output_dir, 'train.log')
        config_logger(log_level=log_level, log_file=log_file)

    from steml.utils import config_gpu
    if gpu_config is not None:
        config_gpu(gpu_config)

    from steml.data.dataset import make_dataset
    from tensorflow.keras.models import load_model
    from steml.models import make_resnet18, get_callbacks

    train_df = pd.read_csv(train_csv)
    train_ds = make_dataset(
        paths=train_df['path'],
        labels=train_df[label],
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        cache=cache,
        shuffle=shuffle_train,
        augment=augment_train,
        balance=balance_train,
    )

    val_ds = None
    if val_csv is not None:
        val_df = pd.read_csv(val_csv)
        val_ds = make_dataset(
            paths=val_df['path'],
            labels=val_df[label],
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
            cache=cache,
            shuffle=shuffle_val,
            augment=augment_val,
            balance=balance_val,
        )

    model = make_resnet18(
        input_shape=(SIZE, SIZE, 3),
        num_classes=num_classes,
        activation=activation,
        lr=lr,
        loss=loss,
        metrics=metrics,
    )

    model_file = os.path.join(output_dir, 'model.h5')
    logging.info(f'Saving model to {model_file}')
    history = model.fit(
        x=train_ds,
        epochs=epochs,
        verbose=1,
        validation_data=val_ds,  # can be None
        callbacks=get_callbacks(
            model_file=model_file,
            epochs=epochs,
            patience=patience,
            lr=lr,
            lr_reduction=lr_reduction,
            lr_patience=lr_patience,
            num_reductions=num_reductions,
            monitor=callback_monitor,
        ),
    )
    history = pd.DataFrame(history.history)
    logging.info(f'Trained model for {len(history)} epochs')

    history.index.name = 'epoch'
    history_file = os.path.join(output_dir, 'history.csv')
    history.to_csv(history_file)
    logging.info(f'Saved training history to {history_file}')

    if test_csv is not None:
        model = load_model(model_file)
        logging.info(f'Loaded best model from {model_file}')
        test_df = pd.read_csv(test_csv)
        test_ds = make_dataset(
            paths=test_df['path'],
            labels=test_df[label],
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
            cache=False,
            shuffle=False,
            augment=False,
            balance=False,
        )
        y_pred_test = model.predict(test_ds)[:, 1]
        predictions_file = os.path.join(output_dir, 'test_predictions.csv')
        pd.DataFrame({'path': test_df['path'], label: y_pred_test}).to_csv(predictions_file, index=False)
        logging.info(f'Generated test predictions to {predictions_file}')


def train_random_splits(
    num_splits: int,
    label: str,
    num_classes: int,
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
    gpu_config: Tuple[int, int],
    cache: bool = False,
    shuffle_train: bool = True,
    shuffle_val: bool = True,
    augment_train: bool = False,
    augment_val: bool = False,
    log_level: LogLevel = LogLevel.INFO,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train_random_splits.log')
    config_logger(log_level=log_level, log_file=log_file)

    # setup trials
    from steml.data import get_tile_paths_labels, get_random_splits
    paths, labels = get_tile_paths_labels(input_dir=input_dir, label=label)
    splits = get_random_splits(num_splits=num_splits, paths=paths, labels=labels)
    digits = len(str(num_splits - 1))  # for 0 padding trial subdirectory names
    for trial, (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    ) in enumerate(splits):
        trial_dir = os.path.join(output_dir, '{trial:0{digits}d}'.format(trial=trial, digits=digits))
        os.makedirs(trial_dir, exist_ok=True)
        train_csv = os.path.join(trial_dir, 'train.csv')
        val_csv = os.path.join(trial_dir, 'val.csv')
        test_csv = os.path.join(trial_dir, 'test.csv')
        pd.DataFrame({'path': train_paths, label: train_labels}).to_csv(train_csv, index=False)
        pd.DataFrame({'path': val_paths, label: val_labels}).to_csv(val_csv, index=False)
        pd.DataFrame({'path': test_paths, label: test_labels}).to_csv(test_csv, index=False)

    # run all trials
    for trial in range(num_splits):
        trial_dir = os.path.join(output_dir, '{trial:0{digits}d}'.format(trial=trial, digits=digits))
        train_csv = os.path.join(trial_dir, 'train.csv')
        val_csv = os.path.join(trial_dir, 'val.csv')
        test_csv = os.path.join(trial_dir, 'test.csv')
        logging.info(f'Running Trial {trial}')
        p = Process(
            target=train,
            name=f'Trial {trial}',
            kwargs={
                'label': label,
                'num_classes': num_classes,
                'train_csv': train_csv,
                'val_csv': val_csv,
                'test_csv': test_csv,
                'cache': cache,
                'shuffle_train': shuffle_train,
                'augment_train': augment_train,
                'shuffle_val': shuffle_val,
                'augment_val': augment_val,
                'activation': activation,
                'batch_size': batch_size,
                'epochs': epochs,
                'patience': patience,
                'lr': lr,
                'lr_reduction': lr_reduction,
                'lr_patience': lr_patience,
                'callback_monitor': 'val_loss',
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


def train_leave_one_out(
    label: str,
    num_classes: int,
    num_inner_splits: int,
    input_dir: str,
    activation: str,
    batch_size: int,
    epochs: int,
    lr: float,
    lr_reduction: Optional[float],
    num_reductions: Optional[int],
    loss: str,
    metrics: List[str],
    output_dir: str,
    num_workers: int,
    gpu_config: Tuple[int, int],
    cache: bool = False,
    shuffle_train: bool = True,
    augment_train: bool = False,
    log_level: LogLevel = LogLevel.INFO,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train_leave_one_out.log')
    config_logger(log_level=log_level, log_file=log_file)

    # load paths/labels
    paths = defaultdict(dict)
    for slide in ['C.diff', 'H.pylori']:
        for section in ['A1', 'B1', 'C1', 'D1']:
            root = os.path.join(input_dir, slide, section)
            df = pd.read_csv(os.path.join(root, f'{label}.csv'))
            df['path'] = root + '/' + df['barcode'] + '.png'
            paths[slide][section] = df

    # leave one out
    for slide in ['H.pylori', 'C.diff']:
        for section in ['A1', 'B1', 'C1', 'D1']:
            train_df = pd.concat([df for k1, v in paths.items() for k2, df in v.items() if k1 != slide and k2 != section], copy=False)
            test_df = paths[slide][section]

            # nested crossval to determine best number of epochs
            logging.info(f'Finding optimal number of epochs by {num_inner_splits}-fold cross val with {slide}/{section} holdout')
            skf = StratifiedKFold(n_splits=num_inner_splits, shuffle=True, random_state=2022)
            inner_splits = []
            for inner_train_idx, inner_test_idx in skf.split(train_df['path'], train_df[label]):
                inner_train_df = train_df.iloc[inner_train_idx]
                inner_test_df = train_df.iloc[inner_test_idx]
                inner_splits.append((inner_train_df, inner_test_df))
            trial_bests = []
            for trial, (inner_train_df, inner_test_df) in enumerate(inner_splits):
                trial_dir = os.path.join(output_dir, slide, section, str(trial))
                os.makedirs(trial_dir, exist_ok=True)
                train_csv = os.path.join(trial_dir, 'train.csv')
                test_csv = os.path.join(trial_dir, 'test.csv')
                inner_train_df.to_csv(train_csv, index=False)
                inner_test_df.to_csv(test_csv, index=False)
                logging.info(f'Running inner trial {trial}')
                logging.info(f'VAL LOSS IS TEST SET')
                p = Process(
                    target=train,
                    name=f'Trial {trial}',
                    kwargs={
                        'label': label,
                        'num_classes': num_classes,
                        'train_csv': train_csv,
                        'val_csv': test_csv,
                        'cache': cache,
                        'shuffle_train': shuffle_train,
                        'augment_train': augment_train,
                        'shuffle_val': False,
                        'augment_val': False,
                        'callback_monitor': 'loss',
                        'activation': activation,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'lr': lr,
                        'lr_reduction': lr_reduction,
                        'num_reductions': num_reductions,
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
                history = pd.read_csv(os.path.join(trial_dir, 'history.csv'))
                trial_best = np.argmin(history['val_loss']) + 1
                trial_bests.append(trial_best)
                logging.info(f'Inner trial {trial} best epoch {trial_best}')
            best_epoch = np.round(np.mean(trial_bests)).astype(int)
            logging.info(f'Holdout {slide}/{section} average best epoch from inner folds {best_epoch}')

            # run outer fold using best number of epochs
            trial_dir = os.path.join(output_dir, slide, section)
            os.makedirs(trial_dir, exist_ok=True)
            train_csv = os.path.join(trial_dir, 'train.csv')
            test_csv = os.path.join(trial_dir, 'test.csv')
            train_df.to_csv(train_csv, index=False)
            test_df.to_csv(test_csv, index=False)
            logging.info(f'Running outer trial with {slide}/{section} holdout')
            p = Process(
                target=train,
                name=f'Trial {slide}/{section}',
                kwargs={
                    'label': label,
                    'num_classes': num_classes,
                    'train_csv': train_csv,
                    'test_csv': test_csv,
                    'cache': cache,
                    'shuffle_train': shuffle_train,
                    'augment_train': augment_train,
                    'callback_monitor': 'loss',
                    'activation': activation,
                    'batch_size': batch_size,
                    'epochs': best_epoch,
                    'lr': lr,
                    'lr_reduction': lr_reduction,
                    'num_reductions': num_reductions,
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


def dispatch_next_job(
    gpu_jobs: List[Tuple[GPU_CONFIG, Optional[Process]]],
    start_next_job: Callable[[GPU_CONFIG], Process],
    refresh: float = 0.5
) -> List[Tuple[GPU_CONFIG, Optional[Process]]]:
    while True:
        for idx, (gpu_config, job) in enumerate(gpu_jobs):
            if job is None:
                job = start_next_job(gpu_config)
                gpu_jobs[idx] = (gpu_config, job)
                return gpu_jobs
            else:
                if not job.is_alive():
                    job.join()
                    job = start_next_job(gpu_config)
                    gpu_jobs[idx] = (gpu_config, job)
                    return gpu_jobs
        sleep(refresh)


def train_nested_leave_one_out(
    label: str,
    num_classes: int,
    input_dir: str,
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
    multi_gpu_config: List[Tuple[int, int]],
    cache: bool = False,
    shuffle_train: bool = True,
    augment_train: bool = False,
    balance_train: bool = False,
    shuffle_val: bool = True,
    augment_val: bool = False,
    balance_val: bool = False,
    log_level: LogLevel = LogLevel.INFO,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train_nested_leave_one_out.log')
    config_logger(log_level=log_level, log_file=log_file)

    # load paths/labels
    samples = []
    for slide in ['C.diff', 'H.pylori']:
        for section in ['A1', 'B1', 'C1', 'D1']:
            root = os.path.join(input_dir, slide, section)
            df = pd.read_csv(os.path.join(root, f'{label}.csv'))
            df['path'] = root + '/' + df['barcode'] + '.png'
            samples.append((f'{slide}.{section}', df))

    gpu_jobs = [(gpu_config, None) for gpu_config in multi_gpu_config]
    for test_sample, test_df in samples:
        for val_sample, val_df in samples:
            if val_sample == test_sample:
                continue
            train_df = pd.concat([df for sample, df in samples if sample not in {test_sample, val_sample}])
            trial_dir = os.path.join(output_dir, test_sample, val_sample)
            os.makedirs(trial_dir, exist_ok=True)
            train_csv = os.path.join(trial_dir, 'train.csv')
            val_csv = os.path.join(trial_dir, 'val.csv')
            test_csv = os.path.join(trial_dir, 'test.csv')
            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)
            test_df.to_csv(test_csv, index=False)
            def start_next_job(gpu_config: GPU_CONFIG) -> Process:
                p = Process(
                    target=train,
                    name=f'Test: {test_sample}; Val: {val_sample}',
                    kwargs={
                        'label': label,
                        'num_classes': num_classes,
                        'train_csv': train_csv,
                        'val_csv': val_csv,
                        'test_csv': test_csv,
                        'cache': cache,
                        'shuffle_train': shuffle_train,
                        'augment_train': augment_train,
                        'balance_train': balance_train,
                        'shuffle_val': shuffle_val,
                        'augment_val': augment_val,
                        'balance_val': balance_val,
                        'callback_monitor': 'val_loss',
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
                return p
            gpu_jobs = dispatch_next_job(gpu_jobs, start_next_job)
            logging.info(f'Training with Test: {test_sample}; Val: {val_sample}')
