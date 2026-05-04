from luxia.photonic_meshes.mesh import PhotonicMesh
from .braided_layer import BraidedLayerT1, BraidedLayerT2
from typing import Optional, List
from luxia.params import DacParams
import torch


class BraidedMesh(PhotonicMesh):
    def __init__(self,
                 n_inputs: int,
                 n_layers: int,
                 mmis_params: Optional[List] = None,
                 cross_params: Optional[List] = None,
                 is_matmul: bool = False,
                 dac_params: Optional[DacParams] = None,
                 debug: bool = False
                 ):
        """
        Create a Clements Mesh with n_layers and n_inputs
        Args:
            n_inputs: number of inputs of the mesh
            n_layers: number of layers of the mesh
            mmi_i_losses: mmis losses of the mesh (list of tensors)
            mmi_imbalances: mmis imbalances of the mesh (list of tensors)
            is_matmul: if the mesh uses the matmul method
            bit_width: bit resolution of the phase shifters of the heaters
            debug: if the mesh is in debug mode, print messages
        """
        list_layers = []
        for i in range(n_layers):
            p = None if mmis_params is None else mmis_params[i]
            c = None if cross_params is None else cross_params[i]
            if i % 2 == 0:
                # Even layer
                layer = BraidedLayerT1(
                    n_inputs=n_inputs,
                    mmis_params=p,
                    dac_params=dac_params,
                    debug=debug
                )
            else:
                # Odd layer
                layer = BraidedLayerT2(
                    n_inputs=n_inputs,
                    cross_params=c[0] if c is not None else None,
                    hf_cross_params=c[1] if c is not None else None,
                    debug=debug
                )
            list_layers.append(layer)

        super(BraidedMesh, self).__init__(
            n_layers=n_layers,
            n_inputs=n_inputs,
            is_matmul=is_matmul,
            layer_list=list_layers,
            debug=debug
        )

    # Quantization error methods are only present in even layers in the mesh
    def get_heaters_quantization_error(self):
        quantization_error = []
        for i, lyr in enumerate(self.get_photonic_layers()):
            if lyr.get_is_even_layer():
                q_error_heater, q_error_mzi = lyr.get_heaters_quantization_error()
                quantization_error.append([q_error_heater, q_error_mzi])
        return quantization_error

    def get_heaters_snr(self):
        lst_snr = []
        for i, lyr in enumerate(self.get_photonic_layers()):
            if lyr.get_is_even_layer():
                lst_snr.append(lyr.get_heaters_snr())

        snr = torch.stack(lst_snr, dim=0).mean(dim=0)
        return snr

    def get_heaters_sqnr(self):
        lst_sqnr = []
        for i, lyr in enumerate(self.get_photonic_layers()):
            if lyr.get_is_even_layer():
                lst_sqnr.append(lyr.get_heaters_sqnr())

        sqnr = torch.stack(lst_sqnr, dim=0).mean(dim=0)
        return sqnr
