import torch


def init_model(
    name: str,
    device: torch.device,
    n_inputs: int,
    n_classes: int,
    n_layers: int = None,
    n_networks: int = 2,
    capacity: int = None,
    mmi_i_losses: torch.Tensor = None,
    mmi_imbalances: torch.Tensor = None,
    is_fastforward: bool = False
) -> torch.nn.Module:
    """
    Initialize the model based on the name provided

    Args:
        name (str): The name of the model to be initialized
        n_inputs (int): the number of inputs of the photonic network
        n_classes (int): The number of classes of the dataset
        n_layers (int): The number of layers
        n_networks (int): The number of photonic networks
        capacity (int): The capacity needed only for the photontorch model
        mmi_i_losses (torch.Tensor): losses of mmis or cross within the network
        mmi_imbalances (torch.Tensor): imbalances of mmi network
        is_fastforward (bool): The fastforward parameter that is used in the
        photonic model, it indicates if the model is fastforward or the
        traditional matrix multiplication

    Returns:
        model (torch.nn.Module): The model
    """

    if name == 'fldzhyan' or name == 'fldzhyan_bell' or name == 'clements' or name == 'clements_bell':
        from photonic_models.photonic_model import PhotonicModel
        return PhotonicModel(architecture_name=name,
                             n_inputs=n_inputs,
                             n_layers=n_layers,
                             n_classes=n_classes,
                             n_networks=n_networks,
                             mmi_i_losses=mmi_i_losses,
                             mmi_imbalances=mmi_imbalances,
                             is_fastforward=is_fastforward,
                             device=device
                             )
    elif name == 'pt_clements':
        from pt_models.pt_clements_model import PtClementsModel
        if capacity is None:
            raise ValueError('Capacity is required for the PtClementsModel')
        m = PtClementsModel(device=device,
                            n_inputs=n_inputs,
                            n_classes=n_classes,
                            capacity=capacity,
                            n_networks=n_networks
                            )
        return m
    elif name == 'nx_rec_mesh':
        from nx_models.nx_rec_mesh_model import NxRecMeshModel
        m = NxRecMeshModel(device=device,
                           n_inputs=n_inputs,
                           n_layers=n_layers,
                           n_classes=n_classes,
                           capacity=capacity,
                           n_networks=n_networks
                           )
        return m
    else:
        raise ValueError('Unknown model: {}'.format(name))
