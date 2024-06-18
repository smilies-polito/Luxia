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
        # Assuming 'info' is an object with attributes 'free', 'used', and 'total' in bytes.
        free_mb = info.free / 1024 ** 2
        used_mb = info.used / 1024 ** 2
        total_mb = info.total / 1024 ** 2

        # Determine the width needed for alignment
        width = max(len(f"{free_mb:.2f}"), len(
            f"{used_mb:.2f}"), len(f"{total_mb:.2f}"))

        # Print formatted strings with aligned decimal points
        print(f"GPU Memory - Used : {used_mb:>{width}.2f} MB")
        print(f"GPU Memory - Free : {free_mb:>{width}.2f} MB")
        print(f"GPU Memory - Total: {total_mb:>{width}.2f} MB")
        nvmlShutdown()
