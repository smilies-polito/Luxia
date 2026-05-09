import torch


def get_tensor_imbalances(
        imbalance: float,
        is_rand: bool,
        model: str,
        n_networks: int,
        n_layers: int,
        n_inputs: int,
        verbose: bool
):
    if verbose:
        print("[*] Loading Fix MMI imbalances")

    if model == 'clements' or model == 'clements_bell':
        if is_rand:
            from mmi_imbalances import get_mzi_rand_imbalances
            mmi_imbalances = get_mzi_rand_imbalances(
                n_networks, n_layers, n_inputs, imbalance)
        else:
            from mmi_imbalances import get_mzi_fix_imbalances
            mmi_imbalances = get_mzi_fix_imbalances(
                n_networks, n_layers, n_inputs, imbalance)
    elif model == 'fldzhyan' or model == 'fldzhyan_bell':
        if is_rand:
            from mmi_imbalances import get_mmi_rand_imbalances
            mmi_imbalances = get_mmi_rand_imbalances(
                n_networks, n_layers, n_inputs, imbalance)
        else:
            from mmi_imbalances import get_mmi_fix_imbalances
            mmi_imbalances = get_mmi_fix_imbalances(
                n_networks, n_layers, n_inputs, imbalance)

    if verbose:
        print(" Network 0 Imbalances:")
        print(mmi_imbalances[0])
        print(" Network 1 Imbalances:")
        print(mmi_imbalances[1])

    return mmi_imbalances


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

