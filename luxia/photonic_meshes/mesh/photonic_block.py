import torch
from luxia import LxModule


class PhotonicBlock(LxModule):
    """ Create the basic block model of the architecture, This is the
        abstract Class of the photonic architecture this class has the
        basic structure of the architecture that is repeated in the
        architecture of 2 inputs and 2 outputs, the structure is as follows:

            Fig. 1

            A_0 --+---------+- C_0
                | PHBLOCK |
            B_0 --+---------+- D_0
                B_1 --+---------+- D_1
                        ...
                   B_N --+---------+- D_N

            N = number of blocks

        Args:
        n_blocks: number of clements blocks to create dowsizes
    """

    def __init__(self, n_blocks: int = 1):
        """
        Constructor of the Class PhotonicBlock
        Here it is define the internal structure of the block,
        Example of the internal structure of the block:
        from ..photonic_components.heater import Heater
        self._heater = Heater(n_heaters=n_blocks)
        from ..photonic_components.mzis.single_heater import SingleHeaterMzi
        self._mzi = SingleHeaterMzi(n_mzi=n_blocks)


        Args:
            n_blocks: number of clements blocks to create dowsizes as shown in
            the Fig 1
        """
        super(PhotonicBlock, self).__init__()
        self._n_blocks = n_blocks
        pass

    def forward(self, portsA, portsB) -> torch.Tensor:
        """
        Here it is define the forward pass of the photonic block,
        for example, the forward pass of the block can be defined as:

            def forward(self, portsA, portsB):
                outPs = self._heater(portsA)
                C, D = self._mzi(outPs, portsB)
                return C, D
            return y_complex

        Args:
            portsA (): All inpurs from the port A, this is a tensor of the size
            of the number of blocks defined in the constructor
            portsB (): All inpurs from the port B, this is a tensor of the size
            of the number of blocks defined in the constructor

        Returns:
            C, D: The outputs of the block C and D in a tensor of size of the
            number of blocks defined in the constructor
        """
        pass

    def get_tf_matrix(self):
        """
        Here it is define how to get the transfer function matrix of the
        all the blocks created in the constructor.
        Here it is an example of how construct the transfer function matrix of
        n blocks, Note that in this case the _mzi object has a method
        called get_tf_matrix() and _heater object has a method called
        getPhases():
            def get_tf_matrix(self):
                # Get the phases of the heaters
                heater_phases = self._heater.getPhases()
                # Create a diagonal matrix with the phases of the heaters
                heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(
                        heater_phases)

                # Get the transfer function matrix of the mzi
                mzi_tf_matrix = self._mzi.get_tf_matrix()
                # Multiply the heater diagonal with the mzi transfer function
                tf_matrix = torch.matmul(heater_diag_matrix, mzi_tf_matrix)
            return tf_matrix

        """
        pass
