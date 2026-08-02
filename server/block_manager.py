from typing import List, Callable
from enum import Enum
from distserve.utils import Stage
from serve.requests import Request, BatchRequests
from config import ModelConfig, ParallelConfig, CacheConfig

class BlockLocation(Enum):
    GPU = "gpu"
    CPU = "cpu"

    def __str__(self):
        return self.value


class BlockManager:
    def __init__(self):
        pass