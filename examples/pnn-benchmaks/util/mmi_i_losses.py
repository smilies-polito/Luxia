import torch


def get_tensor_i_losses(
        loss: float,
        is_rand: bool,
        model: str,
        n_networks: int,
        n_layers: int,
        n_inputs: int,
        verbose: bool
        ):
    if verbose:
        print("[*] Loading random MMI insertion losses")

    if model == 'clements' or model == 'clements_bell':
        if is_rand:
            from mmi_i_losses import get_mzi_rand_losses
            mmi_i_losses = get_mzi_rand_losses(
                n_networks, n_layers, n_inputs, loss)
        else:
            from mmi_i_losses import get_mzi_rand_losses
            mmi_i_losses = get_mzi_rand_losses(
                n_networks, n_layers, n_inputs, loss)
    elif model == 'fldzhyan' or model == 'fldzhyan_bell':
        if is_rand:
            from mmi_i_losses import get_mmi_rand_losses
            mmi_i_losses = get_mmi_rand_losses(
                n_networks, n_layers, n_inputs, loss)
        else:
            from mmi_i_losses import get_mmi_rand_losses
            mmi_i_losses = get_mmi_rand_losses(
                n_networks, n_layers, n_inputs, loss)

    if verbose:
        print(" Network 0 MMI Insertion Losses:")
        print(mmi_i_losses[0])
        print(" Network 1 MMI Insertion Losses:")
        print(mmi_i_losses[1])
    return mmi_i_losses


def get_mmi_rand_losses(n_networks, n_layers, n_inputs, val_range):
    n_mmis = n_inputs // 2  # number of mmis in the model (mmi per layer)
    mmi_losses = []
    for i in range(n_networks):
        imb_arch = []
        for j in range(n_layers):
            n_mmis = n_inputs // 2
            if j % 2 != 0:
                n_mmis -= 1
            imb_arch.append(torch.rand(n_mmis)*val_range)
        mmi_losses.append(imb_arch)

    return mmi_losses


def get_mmi_fix_losses(n_networks, n_layers, n_inputs, imb_val):
    n_mmis = n_inputs // 2  # number of mmis in the model (mmi per layer)
    mmi_losses = []
    for i in range(n_networks):
        imb_arch = []
        for j in range(n_layers):
            n_mmis = n_inputs // 2
            if j % 2 != 0:
                n_mmis -= 1
            imb_arch.append(torch.tensor([imb_val]))
        mmi_losses.append(imb_arch)

    return mmi_losses


def get_mzi_rand_losses(n_networks, n_layers, n_inputs, val_range):
    n_mmis = n_inputs // 2  # number of mmis in the model (mmi per layer)
    mmi_losses = []
    for i in range(n_networks):
        imb_arch = []
        for j in range(n_layers):
            imb_lay = []
            n_mmis = n_inputs // 2
            if j % 2 != 0:
                n_mmis -= 1
            imb_lay.append(torch.rand(n_mmis)*val_range)
            imb_lay.append(torch.rand(n_mmis)*val_range)
            imb_arch.append(imb_lay)
        mmi_losses.append(imb_arch)

    return mmi_losses


def get_mzi_fix_losses(n_networks, n_layers, n_inputs, imb_val):
    n_mmis = n_inputs // 2  # number of mmis in the model (mmi per layer)
    mmi_losses = []
    for i in range(n_networks):
        imb_arch = []
        for j in range(n_layers):
            n_mmis = n_inputs // 2
            if j % 2 != 0:
                n_mmis -= 1
            imb_arch.append(torch.tensor([imb_val]))
        mmi_losses.append(imb_arch)
    return mmi_losses

