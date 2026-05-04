from luxia.lx_module import LxModule
from luxia.photonic_components.mzis.single_heater import SingleHeaterMzi
from typing import Optional
import torch


class PhotonicSigmaLayer(LxModule):
    """
    Abstract class that creates a layer of the photonic architecture as shown
    in the following figures depending on the number of inputs and the type of
    layer to be created chosen by the user (even or odd layer)

        A --+---------+- C
            |   MZI   |
          x-+---------+-x

        A --+---------+- C
            |   MZI   |
          x-+---------+-x

        A --+---------+- C
            |   MZI   |
          x-+---------+-x

        A --+---------+- C
            |   MZI   |
          x-+---------+-x



    Args:
      n_inputs: The number of inputs of the layer is the number of MZIs
          """

    def __init__(self,
                 n_inputs: int,
                 mmi_params: Optional = None,
                 dac_params: Optional = None,
                 debug: bool = False):
        super(PhotonicSigmaLayer, self).__init__(debug=debug)
        self._n_inputs = n_inputs
        self._sh_mzi = SingleHeaterMzi(
                n_mzi=n_inputs,
                mmis_params=mmi_params,
                dac_params=dac_params,
                )

    def forward(self, x_complex):
        """
        Create the forward pass of the layer according to the Figure
        """
        # Get Only the upper arm output of the mzi
        y_0, y_1 = self._sh_mzi(x_complex, torch.torch.zeros_like(x_complex))
        y_complex = y_0
        if self._debug:
            print(" --- layer y: ", y_complex)
            print(" --- layer |y|: ", torch.abs(y_complex))
            print(" --- layer sum(|y|): ", torch.sum(torch.abs(y_complex)))
        return y_complex

    def get_tf_matrix(self):
        tf_matrix = self._sh_mzi.get_tf_matrix()

        if self._debug:
            print(" --- layer TM ", tf_matrix)
        return tf_matrix

    # Getters
    def get_single_heater_mzi(self):
        """
        Get the single heater MZI of the layer
        Returns:

        """
        return self._sh_mzi

    def get_quantization_error(self):
        """
        Get the quantization error of the DAC.
        Returns:
            torch.Tensor: Quantization error.
        """
        return self._sh_mzi.get_heater_quantization_error()

    def get_snr(self):
        return self._sh_mzi.get_heater_snr()

    def get_sqnr(self):
        return self._sh_mzi.get_heater_sqnr()
