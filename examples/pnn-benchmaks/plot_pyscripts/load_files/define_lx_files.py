
def define_lx_digits_file_paths():
    # create dictionary to store file paths
    dic_of_files = {}
    clements_bell_files = [
        "lx_clements_bell_64x64_ep200_bs512_lr0.0005_seed1/140525_161119/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_clements_bell_64x64_ep200_bs512_lr0.0005_seed2/140525_161222/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_clements_bell_64x64_ep200_bs512_lr0.0005_seed3/140525_161324/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_clements_bell_64x64_ep200_bs512_lr0.0005_seed4/140525_161427/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_clements_bell_64x64_ep200_bs512_lr0.0005_seed5/140525_161531/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
    ]
    clements_files = [
        "lx_clements_pdb_64x64_ep200_bs512_lr0.0005_seed1/140525_160640/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_clements_pdb_64x64_ep200_bs512_lr0.0005_seed2/140525_160735/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_clements_pdb_64x64_ep200_bs512_lr0.0005_seed3/140525_160829/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_clements_pdb_64x64_ep200_bs512_lr0.0005_seed4/140525_160923/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_clements_pdb_64x64_ep200_bs512_lr0.0005_seed5/140525_161017/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
    ]
    fldzhyan_files = [
        "lx_fldzhyan_64x128_ep200_bs512_lr0.0005_seed1/140525_161643/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_64x128_ep200_bs512_lr0.0005_seed2/140525_161749/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_64x128_ep200_bs512_lr0.0005_seed3/140525_161854/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_64x128_ep200_bs512_lr0.0005_seed4/140525_162001/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_64x128_ep200_bs512_lr0.0005_seed5/140525_162108/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
    ]
    fldzhyan_bell_files = [
        "lx_fldzhyan_bell_64x128_ep200_bs512_lr0.0005_seed1/140525_162221/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_bell_64x128_ep200_bs512_lr0.0005_seed2/140525_162347/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_bell_64x128_ep200_bs512_lr0.0005_seed3/140525_162513/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_bell_64x128_ep200_bs512_lr0.0005_seed4/140525_162641/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_bell_64x128_ep200_bs512_lr0.0005_seed5/140525_162807/lr_0.0005_ep_200_bs_512_dev_cuda_None.csv",
    ]
    dic_of_files["Clements"] = clements_files
    dic_of_files["Clements Bell"] = clements_bell_files
    dic_of_files["Fldzhyan"] = fldzhyan_files
    dic_of_files["Fldzhyan Bell"] = fldzhyan_bell_files
    return dic_of_files

def define_lx_iris_file_paths():
    # create dictionary to store file paths
    dic_of_files = {}
    clements_bell_files = [
        "lx_clements_bell_4x4_ep400_bs8_lr0.001_seed1/190525_173729/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_clements_bell_4x4_ep400_bs8_lr0.001_seed2/190525_173830/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_clements_bell_4x4_ep400_bs8_lr0.001_seed3/190525_173926/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_clements_bell_4x4_ep400_bs8_lr0.001_seed4/190525_174009/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_clements_bell_4x4_ep400_bs8_lr0.001_seed5/190525_174102/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
    ]
    clements_files = [
        "lx_clements_pdb_4x4_ep400_bs8_lr0.001_seed1/190525_172948/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_clements_pdb_4x4_ep400_bs8_lr0.001_seed2/190525_173045/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_clements_pdb_4x4_ep400_bs8_lr0.001_seed3/190525_173142/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_clements_pdb_4x4_ep400_bs8_lr0.001_seed4/190525_173247/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_clements_pdb_4x4_ep400_bs8_lr0.001_seed5/190525_173331/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
    ]
    fldzhyan_files = [
        "lx_fldzhyan_4x8_ep400_bs8_lr0.001_seed1/190525_174327/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_fldzhyan_4x8_ep400_bs8_lr0.001_seed2/190525_174435/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_fldzhyan_4x8_ep400_bs8_lr0.001_seed3/190525_174527/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_fldzhyan_4x8_ep400_bs8_lr0.001_seed4/190525_174647/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_fldzhyan_4x8_ep400_bs8_lr0.001_seed5/190525_174804/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
    ]
    fldzhyan_bell_files = [
        "lx_fldzhyan_bell_4x8_ep400_bs8_lr0.001_seed1/190525_180759/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_fldzhyan_bell_4x8_ep400_bs8_lr0.001_seed2/190525_180918/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_fldzhyan_bell_4x8_ep400_bs8_lr0.001_seed3/190525_181038/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_fldzhyan_bell_4x8_ep400_bs8_lr0.001_seed4/190525_181212/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
        "lx_fldzhyan_bell_4x8_ep400_bs8_lr0.001_seed5/190525_181336/lr_0.001_ep_400_bs_8_dev_cpu_None.csv",
    ]
    dic_of_files["Clements"] = clements_files
    dic_of_files["Clements Bell"] = clements_bell_files
    dic_of_files["Fldzhyan"] = fldzhyan_files
    dic_of_files["Fldzhyan Bell"] = fldzhyan_bell_files
    return dic_of_files

def define_lx_mnist_file_paths():
    # create dictionary to store file paths
    dic_of_files = {}
    clements_bell_files = [
        "lx_clements_bell_784x784_ep150_bs512_lr0.000038_seed1/210525_112746/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_bell_784x784_ep150_bs512_lr0.000038_seed2/210525_203815/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_bell_784x784_ep150_bs512_lr0.000038_seed3/220525_060002/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_bell_784x784_ep150_bs512_lr0.000038_seed4/220525_135216/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_bell_784x784_ep150_bs512_lr0.000038_seed5/220525_200014/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
    ]
    clements_files = [
        "lx_clements_pdb_784x784_ep150_bs512_lr0.000038_seed1/190525_170827/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_pdb_784x784_ep150_bs512_lr0.000038_seed2/200525_010129/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_pdb_784x784_ep150_bs512_lr0.000038_seed3/200525_092433/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_pdb_784x784_ep150_bs512_lr0.000038_seed4/200525_180203/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_clements_pdb_784x784_ep150_bs512_lr0.000038_seed5/210525_022925/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
    ]
    fldzhyan_files = [
        "lx_fldzhyan_784x1568_ep150_bs512_lr0.000038_seed1/230525_015950/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_784x1568_ep150_bs512_lr0.000038_seed2/230525_095421/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_784x1568_ep150_bs512_lr0.000038_seed3/230525_184037/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_784x1568_ep150_bs512_lr0.000038_seed4/240525_035631/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_784x1568_ep150_bs512_lr0.000038_seed5/240525_134201/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
    ]
    fldzhyan_bell_files = [
        "lx_fldzhyan_bell_784x1568_ep150_bs512_lr0.000038_seed1/160625_120820/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_bell_784x1568_ep150_bs512_lr0.000038_seed2/160625_201645/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_bell_784x1568_ep150_bs512_lr0.000038_seed3/170625_042234/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_bell_784x1568_ep150_bs512_lr0.000038_seed4/170625_122901/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
        "lx_fldzhyan_bell_784x1568_ep150_bs512_lr0.000038_seed5/170625_225011/lr_3.8e-05_ep_150_bs_512_dev_cuda_None.csv",
    ]
    dic_of_files["Clements"] = clements_files
    dic_of_files["Clements Bell"] = clements_bell_files
    dic_of_files["Fldzhyan"] = fldzhyan_files
    dic_of_files["Fldzhyan Bell"] = fldzhyan_bell_files
    return dic_of_files

def define_lx_olivetti_file_paths():
    # create dictionary to store file paths
    dic_of_files = {}
    clements_bell_files = [
      "lx_clements_bell_4096x4096_ep400_bs16_lr0.0000015_seed1/270625_120140/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
      "lx_clements_bell_4096x4096_ep400_bs16_lr0.0000015_seed2/270625_221758/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
      "lx_clements_bell_4096x4096_ep400_bs16_lr0.0000015_seed3/280625_083415/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
      "lx_clements_bell_4096x4096_ep400_bs16_lr0.0000015_seed4/280625_185147/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
      "lx_clements_bell_4096x4096_ep400_bs16_lr0.0000015_seed5/290625_050811/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv"
    ]
    clements_files = [
        "lx_clements_pdb_4096x4096_ep400_bs16_lr0.0000015_seed1/200625_142534/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_clements_pdb_4096x4096_ep400_bs16_lr0.0000015_seed2/210625_025439/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_clements_pdb_4096x4096_ep400_bs16_lr0.0000015_seed3/210625_145701/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_clements_pdb_4096x4096_ep400_bs16_lr0.0000015_seed4/220625_090041/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_clements_pdb_4096x4096_ep400_bs16_lr0.0000015_seed5/220625_212315/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
    ]
    fldzhyan_files = [
        "lx_fldzhyan_4096x8196_ep400_bs16_lr0.0000015_seed1/200525_132251/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_fldzhyan_4096x8196_ep400_bs16_lr0.0000015_seed2/210525_195326/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_fldzhyan_4096x8196_ep400_bs16_lr0.0000015_seed3/220525_205208/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_fldzhyan_4096x8196_ep400_bs16_lr0.0000015_seed4/230525_144554/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_fldzhyan_4096x8196_ep400_bs16_lr0.0000015_seed5/240525_085637/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
    ]
    fldzhyan_bell_files = [
        "lx_fldzhyan_bell_4096x8196_ep400_bs16_lr0.0000015_seed1/290625_153639/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_fldzhyan_bell_4096x8196_ep400_bs16_lr0.0000015_seed2/300625_060700/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_fldzhyan_bell_4096x8196_ep400_bs16_lr0.0000015_seed3/300625_203630/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_fldzhyan_bell_4096x8196_ep400_bs16_lr0.0000015_seed4/010725_105923/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
        "lx_fldzhyan_bell_4096x8196_ep400_bs16_lr0.0000015_seed5/020725_012455/lr_1.5e-06_ep_400_bs_16_dev_cuda_None.csv",
    ]
    dic_of_files["Clements"] = clements_files
    dic_of_files["Clements Bell"] = clements_bell_files
    dic_of_files["Fldzhyan"] = fldzhyan_files
    dic_of_files["Fldzhyan Bell"] = fldzhyan_bell_files
    return dic_of_files
