import torch
import torch.nn as nn


class PhotonicArchitecture(nn.Module):
    """Factory class to create the diffent photonic architectures.
    The architectures are:
    - fldzhyan
    - clements
    - fldzhyan_bell
    - clements_bell

    Args:
        name: name of the architecture to create
        n_layers: number of layers of the architecture
        n_inputs: number of inputs of the architecture
        mmi_i_losses: Tensor of insertion losses of the bs0 and bs1
        mmi_imbalances: Tensor of imbalances of the mmi0 and mmi1
        is_matmul: boolean to indicate if the model is fastforward

    """
    @staticmethod
    def create_architecture(
        name: str,
        n_layers: int,
        n_inputs: int,
        mmi_i_losses: [] = None,
        mmi_imbalances: [] = None,
        is_matmul: bool = False
    ):
        if name == 'fldzhyan':
            from .fldzhyan import FldzhyanArchitecture
            return FldzhyanArchitecture(
                n_inputs=n_inputs,
                n_layers=n_layers,
                mmi_i_losses=mmi_i_losses,
                mmi_imbalances=mmi_imbalances,
                is_matmul=is_matmul
            )
        elif name == 'clements':
            from .clements import ClementsArchitecture
            return ClementsArchitecture(
                n_inputs=n_inputs,
                n_layers=n_layers,
                mmi_i_losses=mmi_i_losses,
                mmi_imbalances=mmi_imbalances,
                is_matmul=is_matmul
            )
        elif name == 'fldzhyan_bell':
            from .fldzhyan_bell import FldzhyanBellArchitecture
            return FldzhyanBellArchitecture(
                n_inputs=n_inputs,
                n_layers=n_layers,
                mmi_i_losses=mmi_i_losses,
                mmi_imbalances=mmi_imbalances,
                is_matmul=is_matmul
            )
        elif name == 'clements_bell':
            from .clements_bell import ClementsBellArchitecture
            return ClementsBellArchitecture(
                n_inputs=n_inputs,
                n_layers=n_layers,
                mmi_i_losses=mmi_i_losses,
                mmi_imbalances=mmi_imbalances,
                is_matmul=is_matmul
            )
        else:
            raise ValueError('Unknown model: {}'.format(name))
