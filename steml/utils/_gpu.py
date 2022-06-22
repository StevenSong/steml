from typing import Dict


def config_gpus(gpus: Dict[int, int]) -> None:
    """
    Set available GPUs and memory limits

    Parameters
    ----------
    gpus: <Dict of int to int> Dictionary of GPU index to memory limit in MB.
          If memory limit is -1, do not limit memory for the given GPU.
    """
    import tensorflow as tf
    all_gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([all_gpus[gpu] for gpu in gpus.keys()], 'GPU')
    for gpu, mem in gpus.items():
        if mem == -1:
            continue
        tf.config.set_logical_device_configuration(
            all_gpus[gpu],
            [tf.config.LogicalDeviceConfiguration(memory_limit=mem)],
        )
