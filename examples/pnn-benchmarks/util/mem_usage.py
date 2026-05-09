import psutil
import os
import torch


def get_gpu_mem_usage(verbose: bool = False):
    try:
        from pynvml import nvmlInit, nvmlShutdown, NVMLError
        from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system.")

        nvmlInit()
        device_index = torch.cuda.current_device()
        handle = nvmlDeviceGetHandleByIndex(device_index)
        info = nvmlDeviceGetMemoryInfo(handle)

        memgpu_free_MB = info.free / 1024 ** 2
        memgpu_used_MB = info.used / 1024 ** 2
        memgpu_total_MB = info.total / 1024 ** 2

        if verbose:
            # Align printing by width of largest value
            width = max(len(f"{memgpu_used_MB:.2f}"),
                        len(f"{memgpu_free_MB:.2f}"),
                        len(f"{memgpu_total_MB:.2f}"))
            print(f"GPU Memory - Used : {memgpu_used_MB:>{width}.2f} MB")
            print(f"GPU Memory - Free : {memgpu_free_MB:>{width}.2f} MB")
            print(f"GPU Memory - Total: {memgpu_total_MB:>{width}.2f} MB")

        return memgpu_free_MB, memgpu_used_MB, memgpu_total_MB

    except NVMLError as e:
        print(f"Failed to initialize NVML: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        try:
            nvmlShutdown()
        except:
            pass  # Safeguard against shutdown errors


def get_cpu_mem_usage(verbose: bool = False):
    """
    Returns private process memory (USS) in MB.
    USS is the most accurate metric for PyTorch CPU memory tracking.
    """
    process = psutil.Process(os.getpid())
    memcpu_used_MB = process.memory_full_info().uss / (1024 ** 2)
    
    if verbose:
        print(f"CPU Private Memory (USS): {memcpu_used_MB:.2f} MB")
    return memcpu_used_MB
