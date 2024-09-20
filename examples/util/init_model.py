import torch


def init_model(
    name: str,
    n_inputs: int,
    n_classes: int,
    n_layers: int = None,
    n_networks: int = 2,
    mmi_i_losses: torch.Tensor = None,
    mmi_imbalances: [] = None,
    is_matmul: bool = False
) -> torch.nn.Module:
    """
    Initialize the model based on the name provided

    Args:
        name (str): The name of the model to be initialized
        n_inputs (int): the number of inputs of the photonic network
        n_classes (int): The number of classes of the dataset
        n_layers (int): The number of layers
        n_networks (int): The number of photonic networks
        mmi_i_losses (torch.Tensor): losses of mmis or cross within the network
        mmi_imbalances (torch.Tensor): imbalances of mmi network
        is_matmul (bool): The fastforward parameter that is used in the
        photonic model, it indicates if the model use the matmul or the photonnn
        method for large photonic circuits
    Returns:
        model (torch.nn.Module): The model
    """

    if name == 'fldzhyan':
        from accelerators.fldzhyan_accelerator import FldzhyanAccelerator
        return FldzhyanAccelerator(
                n_inputs=n_inputs,
                n_layers=n_layers,
                n_classes=n_classes,
                n_networks=n_networks,
                mmi_i_losses=mmi_i_losses,
                mmi_imbalances=mmi_imbalances,
                is_matmul=is_matmul,
                )
    elif name == 'fldzhyan_bell':
        from accelerators.fldzhyan_bell_accelerator import FldzhyanBellAccelerator
        return FldzhyanBellAccelerator(
            n_inputs=n_inputs,
            n_layers=n_layers,
            n_classes=n_classes,
            n_networks=n_networks,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances,
            is_matmul=is_matmul,
        )
    elif name == 'clements':
        from accelerators.clements_accelerator import ClementsAccelerator
        return ClementsAccelerator(
            n_inputs=n_inputs,
            n_layers=n_layers,
            n_classes=n_classes,
            n_networks=n_networks,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances,
            is_matmul=is_matmul,
        )
    elif name == 'my_ph_mesh':
        from accelerators.my_ph_mesh_accelerator import MyPhMeshAccelerator
        return MyPhMeshAccelerator(
            n_inputs=n_inputs,
            n_layers=n_layers,
            n_classes=n_classes,
            n_networks=n_networks,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances,
            is_matmul=is_matmul,
        )
    else:
        raise ValueError('Unknown model: {}'.format(name))
