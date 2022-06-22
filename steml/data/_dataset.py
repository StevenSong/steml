import tensorflow as tf
from typing import Tuple, List


def _load_tile(filename: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_png(tf.io.read_file(filename))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def make_dataset(
    paths: List[str],
    labels: List[int],
    num_classes: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    pds = tf.data.Dataset.from_tensor_slices(paths)
    lds = tf.data.Dataset.from_tensor_slices(tf.one_hot(labels, num_classes))
    ds = tf.data.Dataset.zip((pds, lds))
    if shuffle:
        ds = ds.shuffle(len(labels)//10)
    ds = (
        ds
        .map(_load_tile, num_parallel_calls=num_workers)
        .batch(batch_size)
        .prefetch(2)
    )
    return ds
