import torch
from .utils import get_trained_model


def load_model(
    name: str,
    n_inputs: int,
    n_classes: int,
    is_matmul: bool = False,
    dac_resolution: int = None,
    adc_resolution: int = None,
    het_resolution: int = None,
    pth_model: str = None
) -> torch.nn.Module:
    """
    Initialize the model based on the name provided

    Args:
        name (str): The name of the model to be initialized
        n_inputs (int): the number of inputs of the photonic network
        n_classes (int): The number of classes of the dataset
        dac_resolution (int): Bit resolution of the DAC
        adc_resolution (int): Bit resolution of the ADC
        het_resolution (int): Bit resolution of the heaters
    Returns:
        model (torch.nn.Module): The model
    """

    if name == 'clements':
        return get_trained_model(pth_model)
    elif name == 'full_clements':
        return get_trained_model(pth_model)
    elif name == 'full_nt_clements':
        model = get_trained_model(pth_model)
        from accelerators.full_ntrain_clements import FullNTrainClements
        return FullNTrainClements(
            n_inputs=n_inputs,
            n_classes=n_classes,
            dac_resolution=dac_resolution,
            adc_resolution=adc_resolution,
            heaters_resolution=het_resolution,
            accelerator=model
        )
    else:
        raise ValueError('Unknown model: {}'.format(name))
