from time import sleep
from multiprocessing import Process
from typing import List, Tuple, Optional, Callable
from steml.defines import GPU_CONFIG


def setup_jobs(multi_gpu_config: List[Tuple[int, int]]) -> List[Tuple[GPU_CONFIG, Optional[Process]]]:
    return [(gpu_config, None) for gpu_config in multi_gpu_config]


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


def finish_jobs(gpu_jobs: List[Tuple[GPU_CONFIG, Optional[Process]]]) -> None:
    for _, job in gpu_jobs:
        if job is None:
            continue
        job.join()
