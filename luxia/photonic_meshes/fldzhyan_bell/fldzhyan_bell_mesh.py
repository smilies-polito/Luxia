from luxia.photonic_meshes.mesh import PhotonicMesh
from .fldzhyan_layer import FldzhyanLayer
from typing import Optional, List
from luxia.params import DacParams
import torch


class FldzhyanMesh(PhotonicMesh):
    """ Create the photonic architecture of the Fldzhyan

        Fig. 1 Even number of inputs (n_inputs = 6) and n_layers = 8

        |---------------------------- N = 8 -------------------------------|
        A -[w]-- ----------[w]-- ----------[w]-- ----------[w]-- -----------
                X               X               X               X
        B ------ --[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ---
                        X               X               X               X
        A -[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ------- ---
                X               X               X               X
        B ------ --[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ---
                        X               X               X               X
        A -[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ------- ---
                X               X               X               X
        B ------ --------------- --------------- --------------- -----------

        Fig. 2 Odd number of inputs (n_inputs = 7) and n_layers = 8

        |---------------------------- N = 8 -------------------------------|
        A -[w]-- ----------[w]-- ----------[w]-- ----------[w]-- -----------
                X               X               X               x
        B ------ --[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ---
                        X               X               X               X
        A -[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ------- ---
                X               X               X               X
        B ------ --[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ---
                        X               X               X               X
        A -[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ------- ---
                X               X               X               X
        B ------ --[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ---
                        X               X               X               x
        A ------------------------------------------------------------------

        Args:
            n_layers: number of layers of the photonic architecture
            n_inputs: number of inputs of the photonic architecture
            mmi_i_losses: Tensor of all the losses of the mmis in the
            architecture
            mmi_imbalances: Tensor with imbalances of the mmis in the
            architecture
    """

    def __init__(self, n_layers: int, n_inputs: int,
                 mmis_params: Optional[List] = None,
                 dac_params: Optional[DacParams] = None,
                 is_matmul: bool = False
                 ):

        list_layers = []
        for i in range(n_layers):
            p = None if mmis_params is None else mmis_params[i]
            layer = FldzhyanLayer(
                n_inputs=n_inputs,
                is_even_layer=(i % 2 == 0),
                mmi_params=p,
                dac_params=dac_params
            )
            list_layers.append(layer)

        super(FldzhyanMesh, self).__init__(
            n_layers=n_layers,
            n_inputs=n_inputs,
            is_matmul=is_matmul,
            layer_list=list_layers
        )

    def get_heaters_quantization_error(self):
        quantization_error = []
        for i, lyr in enumerate(self.get_photonic_layers()):
            q_error_heater = lyr.get_heaters_quantization_error()
            quantization_error.append([q_error_heater])
        return quantization_error

    def get_heaters_snr(self):
        lst_snr = []
        for i, lyr in enumerate(self.get_photonic_layers()):
            lst_snr.append(lyr.get_heaters_snr())

        snr = torch.stack(lst_snr, dim=0).mean(dim=0)
        return snr

    def get_heaters_sqnr(self):
        lst_sqnr = []
        for i, lyr in enumerate(self.get_photonic_layers()):
            lst_sqnr.append(lyr.get_heaters_sqnr())

        sqnr = torch.stack(lst_sqnr, dim=0).mean(dim=0)
        return sqnr

