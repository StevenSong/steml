import logging
import tensorflow as tf
from typing import List, Optional
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler


def make_lr_scheduler(epochs, initial_lr, lr_reduction, num_reductions):
    epochs_per_round = epochs // num_reductions

    def scheduler(epoch, lr):
        cur_round = epoch // epochs_per_round
        new_lr = initial_lr * tf.math.pow(lr_reduction, cur_round)
        logging.debug(f'Epoch {epoch} Learning Rate {new_lr:.5e}')
        return new_lr

    return scheduler


def get_callbacks(
    model_file: str,
    epochs: int,
    patience: Optional[int],
    lr: float,
    lr_reduction: Optional[float],
    lr_patience: Optional[int],
    num_reductions: Optional[int],
    monitor: str,
    min_delta: float = 0.0001,
) -> List[Callback]:
    if lr_patience is not None and num_reductions is not None:
        raise ValueError(f'Cannot set both lr_patience (ReduceLROnPlateau) and num_reductions (LearningRateScheulder)')
    elif lr_reduction is not None and lr_patience is None and num_reductions is None:
        raise ValueError(f'lr_reduction must be set with lr_patience (ReduceLROnPlateau) or num_reduction (LearningRateScheulder)')

    callbacks = [
        ModelCheckpoint(
            filepath=model_file,
            monitor=monitor,
            save_best_only=monitor == 'val_loss',
            verbose=1,
        ),
    ]
    if patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                min_delta=min_delta,
                verbose=1,
            )
        )
    if lr_patience is not None:
        callbacks.append(
            ReduceLROnPlateau(
                monitor=monitor,
                factor=lr_reduction,
                patience=lr_patience,
                min_delta=min_delta,
                verbose=1,
            )
        )
    if num_reductions is not None:
        callbacks.append(
            LearningRateScheduler(
                schedule=make_lr_scheduler(
                    epochs=epochs,
                    initial_lr=lr,
                    lr_reduction=lr_reduction,
                    num_reductions=num_reductions,
                )
            )
        )
    return callbacks
