import torch
from torch import nn
from photonnn import PnnModule
from photonnn.photonic_components import Heater
from photonnn.util.util_functions import alternating_tensors


class PhotonicBlock(PnnModule):
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


class PhotonicLayer(PnnModule):
    """
    Abstract class that creates a layer of the photonic architecture as shown
    in the following figures depending on the number of inputs and the type of
    layer to be created chosen by the user (even or odd layer)

      Fig. 1 Even number of Inputs
          a) Even Layer:            b) Odd Layer:
        A --+---------+- C       A -----{p}------ C
            | PHBLOCK |
        B --+---------+- D       A --+---------+- C
                                     | PHBLOCK |
        A --+---------+- C       B --+---------+- D
            | PHBLOCK |
        B --+---------+- D       A --+---------+- C
                                     | PHBLOCK |
        A --+---------+- C       B --+---------+- D
            | PHBLOCK |
        B --+---------+- D       B -----{p}------ C

        ========================================================
        Fig. 2 Odd number of Inputs
          a) Even Layer:              b) Odd Layer:
        A --+---------+- C      A ------{p}----- C
            | PHBLOCK |
        B --+---------+- D      A --+---------+- C
                                     | PHBLOCK |
        A --+---------+- C      B --+---------+- D
            | PHBLOCK |
        B --+---------+- D      A --+---------+- C
                                     | PHBLOCK |
        A --+---------+- C      B --+---------+- D
            | PHBLOCK |
        B --+---------+- D      A --+---------+- C
                                     | PHBLOCK |
        A -----{p}------ C      B --+---------+- D

    {p} = optional phase shifter

    Args:
      n_inputs: number of inputs of the layer, from here it can be said if it
      is an even or and odd layer

      is_even_layer: What type of layer to be created

      opt_ps: Optional phase shifter in the waveguide, If True, the layer in
      the figures {p} will be added otherwise it is a direct connection without
      any phase shifter

      mmi_i_losses: Tesor of all the losses of the blocks, each block contains,
      2 tensors of losses since each block uses 2 mmis

      mmi_imbalances: Tesor of all the imbalances of the blocks, each block
      contains, 2 tensors of imbalance since each block uses 2 mmis
    """

    def __init__(self,
                 n_inputs: int,
                 is_even_layer: bool,
                 opt_ps: bool,
                 photonic_block: PhotonicBlock):
        super(PhotonicLayer, self).__init__()
        self._n_inputs = n_inputs
        self._is_even_layer = is_even_layer
        self._opt_ps = opt_ps
        self._is_even_input = (self._n_inputs % 2 == 0)

        # Create the heaters if the optional phase shifter is True
        if self._opt_ps is True:
            # from photonnn.photonic_components import Heater
            if self._is_even_input is True:
                if self._is_even_layer is not True:
                    self._heater = nn.ModuleList([Heater(1), Heater(1)])
            else:
                self._heater = Heater(1)

        self._ph_blocks = photonic_block

        if self._is_even_input is True:
            if self._is_even_layer is True:
                self.forward = self._Even_Even_Forward
            else:
                self.forward = self._Even_Odd_Forward
        else:
            if self._is_even_layer is True:
                self.forward = self._Odd_Even_Forward
            else:
                self.forward = self._Odd_Odd_Forward

    def _Even_Even_Forward(self, x_complex):
        """
        Create the forward pass of the layer according to the Figure 1.A
        """
        x_complex_0 = x_complex[:, ::2]  # Get all A's (x_complex_0)
        x_complex_1 = x_complex[:, 1::2]  # Get all B's (x_complex_1)
        y_0, y_1 = self._ph_blocks(x_complex_0, x_complex_1)  # Get C's and D's
        return alternating_tensors(y_0, y_1)  # Create the output tensor

    def _Even_Odd_Forward(self, x_complex):
        """
        Create the forward pass of the layer according to the Figure 1.B
        """
        y_complex = torch.zeros_like(x_complex)
        if self._opt_ps is True:
            # A_0 -{p}> C_0 (first elements)
            y_complex[:, 0] = self._heater[0](x_complex[:, 0])
            # A_N -{p}> C_N (lasts elements)
            y_complex[:, -1] = self._heater[1](x_complex[:, - 1])
        else:
            y_complex[:, 0] = x_complex[:, 0]  # A_0 -> C_0
            y_complex[:, -1] = x_complex[:, - 1]  # A_N -> C_N

        x_complex = x_complex[:, 1:-1]  # Remove the first and last elements

        x_complex_0 = x_complex[:, ::2]  # Get all B's (x_complex_0)
        x_complex_1 = x_complex[:, 1::2]  # Get all A's (x_complex_1)
        y_0, y_1 = self._ph_blocks(x_complex_0, x_complex_1)
        y_complex[:, 1:-1] = alternating_tensors(y_0, y_1)
        return y_complex

    def _Odd_Even_Forward(self, x_complex):
        """
        Create the forward pass of the layer according to the Figure 2.A
        """
        y_complex = torch.zeros_like(x_complex)
        if self._opt_ps is True:
            # A_N -{p}> C_N (lasts elements)
            y_complex[:, -1] = self._heater(x_complex[:, -1])
        else:
            y_complex[:, -1] = x_complex[:, -1]  # A_N -> C_N

        x_complex = x_complex[:, :-1]  # Remove the last element
        x_complex_0 = x_complex[:, ::2]  # Get all A's (x_complex_0)
        x_complex_1 = x_complex[:, 1::2]  # Get all B's (x_complex_1)
        y_0, y_1 = self._ph_blocks(x_complex_0, x_complex_1)
        y_complex[:, :-1] = alternating_tensors(y_0, y_1)
        return y_complex

    def _Odd_Odd_Forward(self, x_complex):
        """
        Create the forward pass of the layer according to the Figure 2.B
        """
        y_complex = torch.zeros_like(x_complex)
        if self._opt_ps is True:
            # A_0 -{p[0]}> C_0 (first elements)
            y_complex[:, 0] = self._heater(x_complex[:, 0])
        else:
            y_complex[:, 0] = x_complex[:, 0]  # A_0 -> C_0
        x_complex = x_complex[:, 1:]  # Remove the first elements
        x_complex_0 = x_complex[:, ::2]  # Get all Bs (x_complex_0)
        x_complex_1 = x_complex[:, 1::2]  # Get all As (x_complex_1)
        y_0, y_1 = self._ph_blocks(x_complex_0, x_complex_1)
        y_complex[:, 1:] = alternating_tensors(y_0, y_1)
        return y_complex

    def get_tf_matrix(self):
        if self._is_even_input is True:
            if self._is_even_layer is True:
                tf_matrix = self._ph_blocks.get_tf_matrix()
            else:
                tf_matrix = self._ph_blocks.get_tf_matrix()
                ones = torch.ones(1)
                tf_matrix = torch.block_diag(ones, tf_matrix, ones)
        else:
            if self._is_even_layer is True:
                tf_matrix = self._ph_blocks.get_tf_matrix()
                ones = torch.ones(1)
                tf_matrix = torch.block_diag(tf_matrix, ones)
            else:
                tf_matrix = self._ph_blocks.get_tf_matrix()
                ones = torch.ones(1)
                tf_matrix = torch.block_diag(ones, tf_matrix)
        return tf_matrix


class PhotonicMesh(PnnModule):
    def __init__(self, n_inputs: int, n_layers: int,
                 is_matmul: bool = False,
                 layer_list: [PhotonicLayer] = None
                 ):
        super(PhotonicMesh, self).__init__()
        if len(layer_list) != n_layers:
            raise ValueError(
                "layer_list must have the same number of layers as n_layers")
        if layer_list is None:
            raise ValueError(
                "You must provide a list of PhotonicLayer objects")
        if is_matmul:
            self.forward = self.matmul_forward
        self._ph_layers = nn.Sequential(*layer_list)

    def forward(self, x_complex):
        """
        Args:
            x_complex (): input batch of complex numbers (batch_size, n_inputs)

        Returns:
        """
        return self._ph_layers(x_complex)

    def matmul_forward(self, x_complex):
        """
        Args:
            x_complex (): input batch of complex numbers (batch_size, n_inputs)

        Returns:
        """
        tf_matrix = self.get_tf_matrix()
        x_complex = torch.matmul(x_complex, tf_matrix)
        return x_complex

    def get_tf_matrix(self):
        if len(self._ph_layers) == 1:
            return self._ph_layers[0].get_tf_matrix()
        else:
            tf_matrix = torch.linalg.multi_dot(
                tuple([l.get_tf_matrix() for l in self._ph_layers])
            )

        return tf_matrix
