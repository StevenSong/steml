from typing import List
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def get_callbacks(
    model_file: str,
    patience: int,
    lr_reduction: float,
    lr_patience: int,
    min_delta: float = 0.0001,
) -> List[Callback]:
    return [
        ModelCheckpoint(
            filepath=model_file,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_reduction,
            patience=lr_patience,
            min_delta=min_delta,
            verbose=1,
        ),
    ]
