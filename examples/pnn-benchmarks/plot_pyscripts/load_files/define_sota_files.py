def define_sota_file_paths():
    # create dictionary to store file paths
    dic_of_files = {}
    # Neuroptica implementation files (5 different seeds)
    neu_files = [
        "neu_clements_64x64_ep150_bs512_lr0.0005_seed1/020525_153729/lr_0.0005_ep_150_bs_512_dev_cpu_None.csv",
        "neu_clements_64x64_ep150_bs512_lr0.0005_seed2/020525_153959/lr_0.0005_ep_150_bs_512_dev_cpu_None.csv",
        "neu_clements_64x64_ep150_bs512_lr0.0005_seed3/020525_154229/lr_0.0005_ep_150_bs_512_dev_cpu_None.csv",
        "neu_clements_64x64_ep150_bs512_lr0.0005_seed4/020525_154458/lr_0.0005_ep_150_bs_512_dev_cpu_None.csv",
        "neu_clements_64x64_ep150_bs512_lr0.0005_seed5/020525_154726/lr_0.0005_ep_150_bs_512_dev_cpu_None.csv"
    ]
    # Photontorch implementation files (5 different seeds)
    pt_files = [
        "pt_clements_64x64_ep150_bs512_lr0.0005_seed1/020525_160745/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "pt_clements_64x64_ep150_bs512_lr0.0005_seed2/030525_171429/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "pt_clements_64x64_ep150_bs512_lr0.0005_seed3/040525_181747/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "pt_clements_64x64_ep150_bs512_lr0.0005_seed4/050525_192002/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "pt_clements_64x64_ep150_bs512_lr0.0005_seed5/060525_203146/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv"
    ]

    # Neurophox implementation files (5 different seeds)
    nx_files = [
        "nx_clements_64x64_ep150_bs512_lr0.0005_seed1/020525_170329/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "nx_clements_64x64_ep150_bs512_lr0.0005_seed2/020525_170852/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "nx_clements_64x64_ep150_bs512_lr0.0005_seed3/020525_171457/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "nx_clements_64x64_ep150_bs512_lr0.0005_seed4/020525_172039/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "nx_clements_64x64_ep150_bs512_lr0.0005_seed5/020525_172643/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
    ]
    # LuxIA implementation files (5 different seeds)
    lx_files = [
        "lx_clements_64x64_ep150_bs512_lr0.0005_seed1/020525_153215/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_64x64_ep150_bs512_lr0.0005_seed2/020525_153256/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_64x64_ep150_bs512_lr0.0005_seed3/020525_153336/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_64x64_ep150_bs512_lr0.0005_seed4/020525_153417/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_64x64_ep150_bs512_lr0.0005_seed5/020525_153457/lr_0.0005_ep_150_bs_512_dev_cuda_None.csv"
    ]
    dic_of_files["LuxIA"] = lx_files
    dic_of_files["Photontorch"] = pt_files
    dic_of_files["Neurophox"] = nx_files
    dic_of_files["Neuroptica"] = neu_files
    return dic_of_files
