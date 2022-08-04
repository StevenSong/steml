from enum import Enum
from typing import Tuple


SIZE = 224
GPU_CONFIG = Tuple[int, int]


class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
