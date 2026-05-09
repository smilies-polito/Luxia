from typing import Dict, List, Optional


def define_bs_file_paths() -> Dict[str, Dict[str, Optional[List[str]]]]:
    """
    Define file paths for different libraries and devices with varying batch sizes.
    Returns:
        Dictionary with library names as keys, containing nested dictionaries for CPU and GPU files.
    """
    # Create dictionary to store file paths
    libraries = {
        "LuxIA": {
            "CPU": [
                "lx_clements_800x800_ep1_bs64_lr0.0005/190525_230453/lr_0.0005_ep_1_bs_64_dev_cpu_None.csv",
                "lx_clements_800x800_ep1_bs128_lr0.0005/190525_230541/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "lx_clements_800x800_ep1_bs256_lr0.0005/190525_230616/lr_0.0005_ep_1_bs_256_dev_cpu_None.csv",
                "lx_clements_800x800_ep1_bs512_lr0.0005/190525_230651/lr_0.0005_ep_1_bs_512_dev_cpu_None.csv"
            ],
            "GPU": [
                "lx_clements_800x800_ep1_bs64_lr0.0005/190525_230809/lr_0.0005_ep_1_bs_64_dev_cuda_None.csv",
                "lx_clements_800x800_ep1_bs128_lr0.0005/190525_230833/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
                "lx_clements_800x800_ep1_bs256_lr0.0005/190525_230849/lr_0.0005_ep_1_bs_256_dev_cuda_None.csv",
                "lx_clements_800x800_ep1_bs512_lr0.0005/190525_230901/lr_0.0005_ep_1_bs_512_dev_cuda_None.csv"
            ]
        },
        "Photontorch": {
            "CPU": None,
            "GPU": None
        },
        "Neurophox": {
            "CPU": [
                "nx_clements_800x800_ep1_bs64_lr0.0005/190525_230921/lr_0.0005_ep_1_bs_64_dev_cpu_None.csv",
                "nx_clements_800x800_ep1_bs128_lr0.0005/190525_231026/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "nx_clements_800x800_ep1_bs256_lr0.0005/190525_231110/lr_0.0005_ep_1_bs_256_dev_cpu_None.csv",
                "nx_clements_800x800_ep1_bs512_lr0.0005/190525_231143/lr_0.0005_ep_1_bs_512_dev_cpu_None.csv",
            ],
            "GPU": [
                "nx_clements_800x800_ep1_bs64_lr0.0005/190525_231253/lr_0.0005_ep_1_bs_64_dev_cuda_None.csv",
                "nx_clements_800x800_ep1_bs128_lr0.0005/190525_231324/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
                "nx_clements_800x800_ep1_bs256_lr0.0005/190525_231346/lr_0.0005_ep_1_bs_256_dev_cuda_None.csv",
                "nx_clements_800x800_ep1_bs512_lr0.0005/190525_231404/lr_0.0005_ep_1_bs_512_dev_cuda_None.csv",
            ]
        },
        "Neuroptica": {
            "CPU": [
                "neu_clements_800x800_ep1_bs64_lr0.0005/190525_232423/lr_0.0005_ep_1_bs_64_dev_cpu_None.csv",
                "neu_clements_800x800_ep1_bs128_lr0.0005/200525_045652/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "neu_clements_800x800_ep1_bs256_lr0.0005/200525_054048/lr_0.0005_ep_1_bs_256_dev_cpu_None.csv",
                "neu_clements_800x800_ep1_bs512_lr0.0005/200525_060508/lr_0.0005_ep_1_bs_512_dev_cpu_None.csv",
            ],
            "GPU": None
        }
    }

    return libraries


def define_ns_file_paths() -> Dict[str, Dict[str, Optional[List[str]]]]:
    libraries = {
        "LuxIA": {
            "CPU": [
                "lx_clements_100x100_ep1_bs128_lr0.0005/190525_231444/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "lx_clements_300x300_ep1_bs128_lr0.0005/190525_231450/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "lx_clements_500x500_ep1_bs128_lr0.0005/190525_231502/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "lx_clements_700x700_ep1_bs128_lr0.0005/190525_231522/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "lx_clements_900x900_ep1_bs128_lr0.0005/190525_231551/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
            ],
            "GPU": [
                "lx_clements_100x100_ep1_bs128_lr0.0005/190525_231648/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
                "lx_clements_300x300_ep1_bs128_lr0.0005/190525_231655/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
                "lx_clements_500x500_ep1_bs128_lr0.0005/190525_231704/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
                "lx_clements_700x700_ep1_bs128_lr0.0005/190525_231716/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
                "lx_clements_900x900_ep1_bs128_lr0.0005/190525_231731/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",

            ]
        },
        "Photontorch": {
            "CPU": [
                "pt_clements_100x100_ep1_bs128_lr0.00065/170225_061127/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                None,
                None,
                None,
                None
                ],
            "GPU": None
        },
        "Neurophox": {
            "CPU": [
                "nx_clements_100x100_ep1_bs128_lr0.0005/190525_231750/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "nx_clements_300x300_ep1_bs128_lr0.0005/190525_231758/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "nx_clements_500x500_ep1_bs128_lr0.0005/190525_231810/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "nx_clements_700x700_ep1_bs128_lr0.0005/190525_231832/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "nx_clements_900x900_ep1_bs128_lr0.0005/190525_231912/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
            ],
            "GPU": [
                "nx_clements_100x100_ep1_bs128_lr0.0005/190525_232005/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
                "nx_clements_300x300_ep1_bs128_lr0.0005/190525_232013/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
                "nx_clements_500x500_ep1_bs128_lr0.0005/190525_232023/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
                "nx_clements_700x700_ep1_bs128_lr0.0005/190525_232036/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
                "nx_clements_900x900_ep1_bs128_lr0.0005/190525_232054/lr_0.0005_ep_1_bs_128_dev_cuda_None.csv",
            ]
        },
        "Neuroptica": {
            "CPU": [
                "neu_clements_100x100_ep1_bs128_lr0.0005/200525_062922/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "neu_clements_300x300_ep1_bs128_lr0.0005/200525_063000/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "neu_clements_500x500_ep1_bs128_lr0.0005/200525_063150/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "neu_clements_700x700_ep1_bs128_lr0.0005/200525_064035/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
                "neu_clements_900x900_ep1_bs128_lr0.0005/200525_070903/lr_0.0005_ep_1_bs_128_dev_cpu_None.csv",
            ],
            "GPU": None
        }
    }

    return libraries
