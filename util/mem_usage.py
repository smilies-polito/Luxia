import torch
import psutil
import os


def get_mem_usage(device: torch.device):
    if device.type == 'cuda':
        try:
            from pynvml import nvmlInit, nvmlShutdown, nvmlError
            from pynvml import nvmlDeviceGetHandleByIndex
            from pynvml import nvmlDeviceGetMemoryInfo

            nvmlInit()
            device_index = device.index if device.index is not None else 0
            handle = nvmlDeviceGetHandleByIndex(device_index)
            info = nvmlDeviceGetMemoryInfo(handle)

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

        except nvmlError as e:
            print(f"Failed to initialize NVML: {e}")
        finally:
            nvmlShutdown()
    # CPU Memory
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"CPU Memory - RSS: {memory_info.rss / (1024 ** 2)} MB")
