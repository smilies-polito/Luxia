import torch
from luxia import LxModule
from luxia.params import BsHalfCrossParams
import copy


class HalfCross(LxModule):

    """
    Two input - two output cross beamsplitter.
        The model follows the following S-matrix:

        A --
            X
        - -- --D

        Args:
            n_crosses: number of Cross beamsplitters, to set one after the
            other downsides
            i_loss: Tesor of size n_crosses, this is required for setting the
            insertion loss
            crosstalk: Tensor of size n_crosses, this term is only valid for
            cross type beamsplitter
    """

    def __init__(self, half_cross_params: BsHalfCrossParams):
        # Constructor of the Class Cross
        super(HalfCross, self).__init__()
        self._params = copy.deepcopy(half_cross_params)
        one = torch.tensor([1+0j])

        i_loss = self._params.i_loss
        mC = one if i_loss is None else (1-i_loss)*one
        mC = torch.sqrt(mC)

        crosstalk = self._params.crosstalk
        self.register_buffer(
            '_C', mC if crosstalk is None else mC * (torch.sqrt(crosstalk)))

    def forward(self, portA):

        y = self._C * portA

        return y

    def get_coefficients(self):
        return self._C
