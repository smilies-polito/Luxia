import torch
from luxia.photonic_components.beamsplitters.beamsplitter import Beamsplitter
from luxia.params import CrossParams


class Cross(Beamsplitter):
    """
    Two input - two output cross beamsplitter.
        The model follows the following S-matrix:

        A -- --C                | 0 | 1 |
            X               1 x |   |   |
        B -- --D                | 1 | 0 |
    """

    def __init__(self, cross_params: CrossParams):
        super(Cross, self).__init__(bs_params=cross_params)

        one = torch.tensor([1+0j])
        zro = torch.tensor([0+0j])

        insertion_loss = self._params.insertion_loss
        crosstalk = self._params.crosstalk

        mC = one if insertion_loss is None else torch.sqrt(insertion_loss)

        ct0 = None if crosstalk is None else mC * torch.sqrt(crosstalk)
        ct1 = None if crosstalk is None else mC * torch.sqrt(1 - crosstalk)

        c00 = zro if crosstalk is None else mC * ct0
        c01 = one if crosstalk is None else mC * ct1
        c10 = one if crosstalk is None else mC * ct1
        c11 = zro if crosstalk is None else mC * ct0

        self.register_coefficients(c00, c01, c10, c11)
