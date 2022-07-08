import numpy as np
import tensorflow as tf
from typing import Tuple, List


def _load_tile(filename: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_png(tf.io.read_file(filename))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def _random_rotate_flip_tile(tile: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    rand = tf.random.uniform((2,), 0, 4, dtype=tf.int32)
    tile = tf.image.rot90(tile, k=rand[0])
    if rand[1] == 2:
        tile = tf.image.flip_left_right(tile)
    elif rand[1] == 3:
        tile = tf.image.flip_up_down(tile)
    return tile, label


def make_dataset(
    paths: List[str],
    labels: List[int],
    num_classes: int,
    batch_size: int,
    num_workers: int,
    cache: bool = False,
    shuffle: bool = True,
    augment: bool = False,
    balance: bool = False,
) -> tf.data.Dataset:
    # cache before shuffling, augmenting, or balancing
    if balance:
        paths = np.array(paths)
        labels = np.array(labels)

        def _per_class_ds(c: int) -> tf.data.Dataset:
            idxs = np.squeeze(np.argwhere(labels == c))
            class_paths = paths[idxs]
            class_labels = labels[idxs]
            class_ds = tf.data.Dataset.from_tensor_slices((class_paths, tf.one_hot(class_labels, 2)))
            if cache:
                class_ds = class_ds.map(_load_tile, num_parallel_calls=num_workers)
                class_ds = class_ds.cache()
                if shuffle:
                    class_ds = class_ds.shuffle(len(class_labels))
            else:
                if shuffle:
                    class_ds = class_ds.shuffle(len(class_labels) // 10)
                class_ds = class_ds.map(_load_tile, num_parallel_calls=num_workers)
            return class_ds
        balanced_ds = tf.data.experimental.sample_from_datasets([
            _per_class_ds(c).repeat()
            for c in range(num_classes)
        ])
        ds = tf.data.Dataset.zip((balanced_ds, tf.data.Dataset.range(len(labels)))).map(lambda dat, _: dat)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, tf.one_hot(labels, num_classes)))
        if cache:
            ds = ds.map(_load_tile, num_parallel_calls=num_workers)
            ds = ds.cache()
            if shuffle:
                ds = ds.shuffle(len(labels))
        else:
            if shuffle:
                ds = ds.shuffle(len(labels) // 10)
            ds = ds.map(_load_tile, num_parallel_calls=num_workers)
    if augment:
        ds = ds.map(_random_rotate_flip_tile, num_parallel_calls=num_workers)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(2)
    return ds
