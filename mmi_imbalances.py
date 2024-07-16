import torch


def get_mmi_rand_imbalances(n_networks, n_layers, n_inputs, val_range):
    n_mmis = n_inputs // 2  # number of mmis in the model (mmi per layer)
    mmi_imbalances = []
    for i in range(n_networks):
        imb_arch = []
        for j in range(n_layers):
            n_mmis = n_inputs // 2
            if j % 2 != 0:
                n_mmis -= 1
            imb_arch.append(torch.rand(n_mmis)*val_range)
        mmi_imbalances.append(imb_arch)

    return mmi_imbalances


def get_mmi_fix_imbalances(n_networks, n_layers, n_inputs, imb_val):
    mmi_imbalances = []
    for i in range(n_networks):
        imb_arch = []
        for j in range(n_layers):
            imb_arch.append(torch.tensor([imb_val]))
        mmi_imbalances.append(imb_arch)

    return mmi_imbalances


def get_mzi_rand_imbalances(n_networks, n_layers, n_inputs, val_range):
    n_mmis = n_inputs // 2  # number of mmis in the model (mmi per layer)
    mmi_imbalances = []
    for i in range(n_networks):
        imb_arch = []
        for j in range(n_layers):
            n_mmis = n_inputs // 2
            if j % 2 != 0:
                n_mmis -= 1
            imb_arch.append(torch.rand(2, n_mmis)*val_range)
        mmi_imbalances.append(imb_arch)

    return mmi_imbalances


def get_mzi_fix_imbalances(n_networks, n_layers, n_inputs, imb_val):
    mmi_imbalances = []
    for i in range(n_networks):
        imb_arch = []
        for j in range(n_layers):
            imb_arch.append(torch.tensor([imb_val]))
        mmi_imbalances.append(imb_arch)
    return mmi_imbalances
