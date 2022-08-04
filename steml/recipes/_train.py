import os
import logging
import numpy as np
import pandas as pd
from multiprocessing import Process
from typing import List, Optional, Tuple, Union
from sklearn.model_selection import StratifiedKFold
from steml.defines import SIZE, GPU_CONFIG, LogLevel
from steml.utils import config_logger, setup_jobs, dispatch_next_job, finish_jobs


def train(
    label: Union[str, List[str]],
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
    continuous: bool = False,
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
    min_delta: float = 0.0001,
    gpu_config: Optional[Tuple[int, int]] = None,
    log_level: LogLevel = LogLevel.INFO,
    skip_log_config: bool = False,
) -> None:
    if not isinstance(label, list):
        label = [label]
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
        continuous=continuous,
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
            continuous=continuous,
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
            min_delta=min_delta,
        ),
    )
    history = pd.DataFrame(history.history)
    logging.info(f'Trained model for {len(history)} epochs')

    history.index.name = 'epoch'
    history_file = os.path.join(output_dir, 'history.csv')
    history.to_csv(history_file)
    logging.info(f'Saved training history to {history_file}')

    model = load_model(model_file)
    logging.info(f'Loaded best model from {model_file}')
    train_df = pd.read_csv(train_csv)
    train_ds = make_dataset(
        paths=train_df['path'],
        labels=train_df[label],
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        cache=False,
        shuffle=False,
        augment=False,
        balance=False,
        continuous=continuous,
    )
    y_pred_train = model.predict(train_ds)
    predictions_file = os.path.join(output_dir, 'train_predictions.csv')
    pd.concat([pd.Series(train_df['path'], name='path'), pd.DataFrame(y_pred_train, columns=label)], axis=1).to_csv(predictions_file, index=False)
    logging.info(f'Generated train predictions to {predictions_file}')

    if test_csv is not None:
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
            continuous=continuous,
        )
        y_pred_test = model.predict(test_ds)
        predictions_file = os.path.join(output_dir, 'test_predictions.csv')
        pd.concat([pd.Series(test_df['path'], name='path'), pd.DataFrame(y_pred_test, columns=label)], axis=1).to_csv(predictions_file, index=False)
        logging.info(f'Generated test predictions to {predictions_file}')


def _normalize_and_save(
    label: Union[str, List[str]],
    train_df: pd.DataFrame,
    train_csv: str,
    scale_csv: str,
    val_df: Optional[pd.DataFrame] = None,
    val_csv: Optional[str] = None,
    test_df: Optional[pd.DataFrame] = None,
    test_csv: Optional[str] = None,
):
    if not isinstance(label, list):
        label = [label]
    mean = train_df[label].mean()
    std = train_df[label].std(ddof=0)
    for df, df_csv in [(train_df, train_csv), (val_df, val_csv), (test_df, test_csv)]:
        if df is not None:
            df = df.copy()
            for l in label:
                df = pd.concat([df, pd.Series(df[l], name=f'{l}_original')], axis=1)
                # df[f'{l}_original'] = df[l]
                df[l] = (df[l] - mean[l]) / std[l]
            df.to_csv(df_csv, index=False)
    scale_df = pd.concat([mean.reset_index()[[0]].T, std.reset_index()[[0]].T])
    scale_df.columns = label
    scale_df.index = ['mean', 'std']
    scale_df.to_csv(scale_csv)


def train_random_splits(
    num_splits: int,
    label: Union[str, List[str]],
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
    continuous: bool = False,
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
        train_df = pd.DataFrame({'path': train_paths, label: train_labels})
        val_df = pd.DataFrame({'path': val_paths, label: val_labels})
        test_df = pd.DataFrame({'path': test_paths, label: test_labels})
        if continuous:
            _normalize_and_save(
                label=label,
                scale_csv=os.path.join(trial_dir, 'scale.csv'),
                train_df=train_df,
                train_csv=train_csv,
                val_df=val_df,
                val_csv=val_csv,
                test_df=test_df,
                test_csv=test_csv,
            )

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
                'continuous': continuous,
            },
        )
        p.start()
        p.join()


def train_loo(
    label: Union[str, List[str]],
    num_classes: int,
    samples: List[Tuple[str, pd.DataFrame]],
    activation: str,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    loss: str,
    metrics: List[str],
    callback_monitor: str,
    min_delta: float,
    output_dir: str,
    num_workers: int,
    multi_gpu_config: List[Tuple[int, int]],
    continuous: bool = False,
    cache: bool = False,
    shuffle_train: bool = True,
    augment_train: bool = False,
    balance_train: bool = False,
    log_level: LogLevel = LogLevel.INFO,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train_nested_leave_one_out.log')
    config_logger(log_level=log_level, log_file=log_file)
    gpu_jobs = setup_jobs(multi_gpu_config)

    for test_sample, test_df in samples:
        train_df = pd.concat([df for sample, df in samples if sample != test_sample])
        trial_dir = os.path.join(output_dir, test_sample)
        os.makedirs(trial_dir, exist_ok=True)
        train_csv = os.path.join(trial_dir, 'train.csv')
        test_csv = os.path.join(trial_dir, 'test.csv')
        if continuous:
            _normalize_and_save(
                label=label,
                scale_csv=os.path.join(trial_dir, 'scale.csv'),
                train_df=train_df,
                train_csv=train_csv,
                test_df=test_df,
                test_csv=test_csv,
            )

        def start_next_job(gpu_config: GPU_CONFIG) -> Process:
            p = Process(
                target=train,
                name=f'Test: {test_sample}',
                kwargs={
                    'label': label,
                    'num_classes': num_classes,
                    'train_csv': train_csv,
                    'test_csv': test_csv,
                    'cache': cache,
                    'shuffle_train': shuffle_train,
                    'augment_train': augment_train,
                    'balance_train': balance_train,
                    'activation': activation,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'patience': patience,
                    'lr': lr,
                    'loss': loss,
                    'metrics': metrics,
                    'min_delta': min_delta,
                    'callback_monitor': callback_monitor,
                    'output_dir': trial_dir,
                    'num_workers': num_workers,
                    'gpu_config': gpu_config,
                    'skip_log_config': True,
                    'continuous': continuous,
                },
            )
            p.start()
            return p
        gpu_jobs = dispatch_next_job(gpu_jobs, start_next_job)
        logging.info(f'Training with Test: {test_sample}')
    finish_jobs(gpu_jobs)


def train_outer_loo_inner_cv(
    label: Union[str, List[str]],
    num_classes: int,
    num_inner_splits: int,
    samples: List[Tuple[str, pd.DataFrame]],
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
    multi_gpu_config: List[Tuple[int, int]],
    continuous: bool = False,
    cache: bool = False,
    shuffle_train: bool = True,
    augment_train: bool = False,
    balance_train: bool = False,
    log_level: LogLevel = LogLevel.INFO,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train_leave_one_out.log')
    config_logger(log_level=log_level, log_file=log_file)
    gpu_jobs = setup_jobs(multi_gpu_config)

    # leave one out
    for test_sample, test_df in samples:
        train_df = pd.concat([df for sample, df in samples if sample != test_sample], copy=False)

        # nested crossval to determine best number of epochs
        logging.info(
            f'Finding optimal number of epochs by {num_inner_splits}-fold cross val with {test_sample} holdout')
        skf = StratifiedKFold(n_splits=num_inner_splits, shuffle=True, random_state=2022)
        inner_splits = []
        for inner_train_idx, inner_test_idx in skf.split(train_df['path'], train_df[label]):
            inner_train_df = train_df.iloc[inner_train_idx]
            inner_test_df = train_df.iloc[inner_test_idx]
            inner_splits.append((inner_train_df, inner_test_df))
        trial_bests = []
        for trial, (inner_train_df, inner_test_df) in enumerate(inner_splits):
            trial_dir = os.path.join(output_dir, test_sample, str(trial))
            os.makedirs(trial_dir, exist_ok=True)
            train_csv = os.path.join(trial_dir, 'train.csv')
            test_csv = os.path.join(trial_dir, 'test.csv')
            if continuous:
                _normalize_and_save(
                    label=label,
                    scale_csv=os.path.join(trial_dir, 'scale.csv'),
                    train_df=inner_train_df,
                    train_csv=train_csv,
                    test_df=inner_test_df,
                    test_csv=test_csv,
                )

            def start_next_job(gpu_config: GPU_CONFIG) -> Process:
                logging.info(f'Running inner trial {test_sample}/{trial}')
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
                        'balance_train': balance_train,
                        'shuffle_val': False,
                        'augment_val': False,
                        'balance_val': False,
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
                        'continuous': continuous,
                    },
                )
                p.start()
                return p

            gpu_jobs = dispatch_next_job(gpu_jobs, start_next_job)
        finish_jobs(gpu_jobs)

        for trial, _ in enumerate(inner_splits):
            trial_dir = os.path.join(output_dir, test_sample, str(trial))
            history = pd.read_csv(os.path.join(trial_dir, 'history.csv'))
            trial_best = np.argmin(history['val_loss']) + 1
            trial_bests.append(trial_best)
            logging.info(f'Inner trial {test_sample}/{trial} best epoch {trial_best}')
        best_epoch = np.round(np.mean(trial_bests)).astype(int)
        logging.info(f'Holdout {test_sample} average best epoch from inner folds {best_epoch}')

        # run outer fold using best number of epochs
        trial_dir = os.path.join(output_dir, test_sample)
        os.makedirs(trial_dir, exist_ok=True)
        train_csv = os.path.join(trial_dir, 'train.csv')
        test_csv = os.path.join(trial_dir, 'test.csv')
        if continuous:
            _normalize_and_save(
                label=label,
                scale_csv=os.path.join(trial_dir, 'scale.csv'),
                train_df=train_df,
                train_csv=train_csv,
                test_df=test_df,
                test_csv=test_csv,
            )
        logging.info(f'Running outer trial with {test_sample} holdout')
        p = Process(
            target=train,
            name=f'Trial {test_sample}',
            kwargs={
                'label': label,
                'num_classes': num_classes,
                'train_csv': train_csv,
                'test_csv': test_csv,
                'cache': cache,
                'shuffle_train': shuffle_train,
                'augment_train': augment_train,
                'balance_train': balance_train,
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
                'gpu_config': multi_gpu_config[0],
                'skip_log_config': True,
                'continuous': continuous,
            },
        )
        p.start()
        p.join()


def train_nested_loo(
    label: Union[str, List[str]],
    num_classes: int,
    samples: List[Tuple[str, pd.DataFrame]],
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
    continuous: bool = False,
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
    gpu_jobs = setup_jobs(multi_gpu_config)

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
            if continuous:
                _normalize_and_save(
                    label=label,
                    scale_csv=os.path.join(trial_dir, 'scale.csv'),
                    train_df=train_df,
                    train_csv=train_csv,
                    val_df=val_df,
                    val_csv=val_csv,
                    test_df=test_df,
                    test_csv=test_csv,
                )

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
                        'continuous': continuous,
                    },
                )
                p.start()
                return p
            gpu_jobs = dispatch_next_job(gpu_jobs, start_next_job)
            logging.info(f'Training with Test: {test_sample}; Val: {val_sample}')
