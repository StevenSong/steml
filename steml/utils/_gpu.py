import logging
from steml.defines import GPU_CONFIG


def config_gpu(gpu_config: GPU_CONFIG) -> None:
    gpu, mem = gpu_config
    if mem == -1:
        logging.debug(f'Configuring all available memory on GPU {gpu}')
    else:
        logging.debug(f'Configuring {mem} MB of memory on GPU {gpu}')
    import tensorflow as tf
    all_gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([all_gpus[gpu]], 'GPU')
    if mem == -1:
        return
    tf.config.set_logical_device_configuration(
        all_gpus[gpu],
        [tf.config.LogicalDeviceConfiguration(memory_limit=mem)],
    )
