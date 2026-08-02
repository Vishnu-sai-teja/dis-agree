from enum import Enum

GB = 1 << 30 
MB = 1 << 20

class Counter:
    def __init__(self, start=0):
        self.counter = start
    
    def __next__(self):
        current_count = self.counter
        self.counter += 1
        return self.current_count
    
    def reset(self):
        self.counter = 0

class Stage(Enum):
    CONTEXT = "contenxt"
    DECODE = "decode"

    def __str__(self):
        return self.value
        