import os
import pandas as pd
from typing import Dict, List
from steml.defines import SIZE


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
) -> None:
    from steml.utils import config_gpus
    config_gpus({gpu: mem for gpu, mem in gpu_config.items()})

    from steml.data import make_dataset
    from steml.models import ResNet18, get_callbacks

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    num_classes = 2

    train_ds = make_dataset(paths=train_df['path'], labels=train_df[label], num_classes=num_classes, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_ds = make_dataset(paths=val_df['path'], labels=val_df[label], num_classes=num_classes, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model = ResNet18(
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

    test_ds = make_dataset(paths=test_df['path'], labels=test_df[label], num_classes=num_classes, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    y_hat_test = model.predict(test_ds)[:, 1]
    pd.DataFrame({'path': test_df['path'], label: y_hat_test}).to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
