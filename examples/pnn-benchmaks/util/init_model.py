import torch


def init_model(
    name: str,
    n_inputs: int,
    n_classes: int,
    n_layers: int = None,
    window_size: int = 1,
    n_networks: int = 2,
    mmi_i_losses: torch.Tensor = None,
    mmi_imbalances: [] = None,
    is_matmul: bool = False,
    dac_resolution: int = None,
    adc_resolution: int = None,
    het_resolution: int = None,
    pth_model: str = None,
    rounding_type: str = None,
    device: torch.device = torch.device('cpu'),  # Valid only for other tools
) -> torch.nn.Module:
    """
    Initialize the model based on the name provided

    Args:
        name (str): The name of the model to be initialized
        n_inputs (int): the number of inputs of the photonic network
        n_classes (int): The number of classes of the dataset
        n_layers (int): The number of layers
        window_size (int): Number of layers per photonic window for slicing. 
                           Default=1 (no windowing). (NEW)
        n_networks (int): The number of photonic networks
        mmi_i_losses (torch.Tensor): losses of mmis or cross within the network
        mmi_imbalances (torch.Tensor): imbalances of mmi network
        dac_resolution (int): Bit resolution of the DAC
        adc_resolution (int): Bit resolution of the ADC
        het_resolution (int): Bit resolution of the heaters
        is_matmul (bool): The fastforward parameter that is used in the
        photonic model, it indicates if the model use the matmul or the photonnn
        method for large photonic circuits
    Returns:
        model (torch.nn.Module): The model
    """

    if name == 'lx_fldzhyan':
        from lx_models.fldzhyan_accelerator import FldzhyanAccelerator
        return FldzhyanAccelerator(
                n_inputs=n_inputs,
                n_layers=n_layers,
                n_classes=n_classes,
                n_networks=n_networks,
                is_matmul=is_matmul,
                )
    elif name == 'lx_fldzhyan_bell':
        from lx_models.fldzhyan_bell_accelerator import FldzhyanBellAccelerator
        return FldzhyanBellAccelerator(
            n_inputs=n_inputs,
            n_layers=n_layers,
            n_classes=n_classes,
            n_networks=n_networks,
            is_matmul=is_matmul,
        )
    elif name == 'lx_clements':
        from lx_models.clements_accelerator import ClementsAccelerator
        return ClementsAccelerator(
            n_inputs=n_inputs,
            n_layers=n_layers,
            window_size=window_size,
            n_classes=n_classes,
            n_networks=n_networks,
            is_matmul=is_matmul,
        )
    elif name == 'lx_clements_bell':
        from lx_models.clements_bell_accelerator import ClementsBellAccelerator
        return ClementsBellAccelerator(
            n_inputs=n_inputs,
            n_layers=n_layers,
            window_size=window_size,
            n_classes=n_classes,
            n_networks=n_networks,
            is_matmul=is_matmul,
        )
    elif name == 'lx_clements_pdb':
        from lx_models.clements_accelerator_pdb import ClementsAcceleratorPDB
        return ClementsAcceleratorPDB(
            n_inputs=n_inputs,
            n_layers=n_layers,
            window_size=window_size,
            n_classes=n_classes,
            n_networks=n_networks,
            is_matmul=is_matmul,
        )
    elif name == 'nx_clements':
        from nx_models.nx_rec_mesh_model import NxRecMeshModel
        return NxRecMeshModel(
            device=device,
            n_inputs=n_inputs,
            n_layers=n_layers,
            n_classes=n_classes,
            n_architectures=n_networks,
        )
    elif name == 'neu_clements':
        from neu_models.neu_clements_model import NeuClementsModel
        return NeuClementsModel(
            n_inputs=n_inputs,
            n_layers=n_layers,
            n_classes=n_classes,
            n_architectures=n_networks,
        )
    elif name == 'pt_clements':
        from pt_models.pt_clements_model import PtClementsModel
        torch.set_default_dtype(torch.float32)
        # torch.set_default_dtype(torch.float16), un comment for lower datatype
        return PtClementsModel(
            device=device,
            n_inputs=n_inputs,
            n_classes=n_classes,
            capacity=n_layers,
            n_networks=n_networks,
        )
    else:
        raise ValueError('Unknown model: {}'.format(name))
