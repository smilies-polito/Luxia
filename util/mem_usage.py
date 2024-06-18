import torch
import psutil
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    from pynvml import *


def print_memory_usage():
    # CPU Memory
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"CPU Memory - RSS: {memory_info.rss / (1024 ** 2)} MB")

    if device.type == 'cuda':
        # Initialize NVML
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # GPU 0
        # GPU Memory
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU Memory - Total: {info.total / 1024 ** 2} MB")
        print(f"GPU Memory - Free: {info.free / 1024 ** 2} MB")
        print(f"GPU Memory - Used: {info.used / 1024 ** 2} MB")
        nvmlShutdown()
