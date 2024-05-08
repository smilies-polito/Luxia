import torch
from torch import nn
from .photonic_components.heater import Heater
from .photonic_components.beamsplitters.mmi import Mmi
from .photonic_components.photodetector import Photodetector


class FldzhyanBlock(nn.Module):
    def __init__(self, n_blocks: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None
                 ):
        """ Create the basic model of the Fldzhyan architecture, the basic
        block is a phase shifter followed by a mmi

           wg0
        A -[w]-- ---C
                X
        B ------ ---D

        Args:
            n_blocks: number of fldzhyzn blocks to create dowsizes
            mmi_i_losses: Tensor of insertion losses of the mmis
            mmi_imbalances: Tensor of the imbalances of the mmis
        """
        super(FldzhyanBlock, self).__init__()
        self._heater = Heater(n_heaters=n_blocks)
        self._mmi = Mmi(n_mmis=n_blocks,
                        i_loss=mmi_i_losses,
                        imbalance=mmi_imbalances)

    def forward(self, portsA, portsB):
        outPs = self._heater(portsA)
        y_0, y_1 = self._mmi(outPs, portsB)

        # return the elements in an alternative form
        y_complex = torch.stack([y_0, y_1], axis=2).view(y_0.shape[0], -1)
        return y_complex

    def getSMatrix(self):
        s_mmi_mats = self._mmi.getSMatrix()
        s_mmi_diag_mat = torch.block_diag(*s_mmi_mats)

        h_phases = self._heater.getPhases()
        v_ones = torch.ones_like(h_phases)
        # Stack the tensors along a new dimension
        h_stack_phases = torch.stack((h_phases, v_ones), dim=1)
        # Reshape the stacked tensor to flatten it
        h_vec_phases = h_stack_phases.view(-1)
        h_mat_diag = torch.diag(h_vec_phases)

        s_block_matrix = torch.matmul(h_mat_diag, s_mmi_diag_mat)
        return s_block_matrix


class FldzhyanLayer(nn.Module):
    """
      Fig. 1 Even number of Inputs
          a) Even Layer:      b) Odd Layer:
        A -[w]-- ---C       A ___________ C
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A -[w]-- ---C       A ------ -----D
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A -[w]-- ---C       A ------ -----D
                X
        B ------ ---D       B ___________ C

        ==========================================
        Fig. 2 Odd f Inputs
          a) Even Layer:    b) Odd Layer:
        A -[w]-- ---C       A ___________ C
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A -[w]-- ---C       A ------ -----D
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A -[w]-- ---C       A ------ -----D
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A _________ C       A ------ -----D
    Args:
        n_inputs: n inputs of the layer, from here it can be said
        if it is an even or and odd layer
        is_even_layer: What type of layer to be created
        mmi_i_losses: Tensor of all the losses of the mmis of the block
        mmi_imbalances: Tensor with imbalances of the mmis of the block
    """

    def __init__(self, n_inputs: int,
                 is_even_layer: bool = True,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None
                 ):
        super(FldzhyanLayer, self).__init__()
        self._n_inputs = n_inputs
        self._is_even_layer = is_even_layer
        self._n_blocks = n_inputs // 2
        self._is_even_input = False

        if self._n_inputs % 2 == 0:
            self._is_even_input = True
            if self._is_even_layer is False:
                self._n_blocks = self._n_blocks - 1

        self._ph_blocks = FldzhyanBlock(
            n_blocks=self._n_blocks,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances
            )

    def forward(self, x_complex):
        y_complex = torch.zeros_like(x_complex)
        if self._is_even_input is True:
            if self._is_even_layer is True:
                # figure 1.a
                # Create a tensor with all As (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all Bs (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex = self._ph_blocks(x_complex_0, x_complex_1)
            else:
                # figure 1.b
                # Pass to the output all the first elements
                y_complex[:, 0] = x_complex[:, 0]
                # Pass to the output all the last elements
                y_complex[:, -1] = x_complex[:, - 1]
                x_complex = x_complex[:, 1:-1]
                # Create a tensor with all Bs (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all As (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, 1:-1] = self._ph_blocks(x_complex_0, x_complex_1)
        else:
            if self._is_even_layer is True:
                # figure 2.a
                # Pass to the output all the last elements
                y_complex[:, -1] = x_complex[:, - 1]
                x_complex = x_complex[:, :-1]
                # Create a tensor with all As (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all Bs (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, :-1] = self._ph_blocks(x_complex_0, x_complex_1)
            else:
                # figure 2.b
                # Pass to the output all the first elements
                y_complex[:, 0] = x_complex[:, 0]
                x_complex = x_complex[:, 1:]
                # Create a tensor with all Bs (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all As (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, 1:] = self._ph_blocks(x_complex_0, x_complex_1)
        return y_complex

    def getSMatrix(self):
        if self._is_even_input is True:
            if self._is_even_layer is True:
                s_matrix = self._ph_blocks.getSMatrix()
            else:
                s_matrix = self._ph_blocks.getSMatrix()
                ones = torch.ones(1)
                s_matrix = torch.block_diag(ones, s_matrix, ones)
        else:
            if self._is_even_layer is True:
                s_matrix = self._ph_blocks.getSMatrix()
                ones = torch.ones(1)
                s_matrix = torch.block_diag(s_matrix, ones)
            else:
                s_matrix = self._ph_blocks.getSMatrix()
                ones = torch.ones(1)
                s_matrix = torch.block_diag(ones, s_matrix)
        return s_matrix


class FldzhyanNetwork(nn.Module):
    """ Create the photonic network of the Fldzhyan architecture

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
            n_layers: number of layers of the photonic network
            n_inputs: number of inputs of the photonic network
            mmi_i_losses: Tensor of all the losses of the mmis in the network
            mmi_imbalances: Tensor with imbalances of the mmis in the network
    """

    def __init__(self, n_layers: int, n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None,
                 ):
        super(FldzhyanNetwork, self).__init__()

        self._ph_layers = nn.ModuleList()

        for i in range(n_layers):
            i_losses = None
            imbalance = None
            if mmi_i_losses is not None:
                i_losses = mmi_i_losses[i]
            if mmi_imbalances is not None:
                imbalance = mmi_imbalances[i]

            if i % 2 == 0:
                self._ph_layers.append(FldzhyanLayer(
                    n_inputs=n_inputs,
                    is_even_layer=True,
                    mmi_i_losses=i_losses,
                    mmi_imbalances=imbalance
                    ))
            else:
                self._ph_layers.append(FldzhyanLayer(
                    n_inputs=n_inputs,
                    is_even_layer=False,
                    mmi_i_losses=i_losses,
                    mmi_imbalances=imbalance,
                    ))

    def forward(self, x_complex):
        for layer in self._ph_layers:
            x_complex = layer(x_complex)
        return x_complex

    def get_SMatrix(self):
        s_matrix = None
        for layer in self._ph_layers:
            if s_matrix is None:
                s_matrix = layer.getSMatrix()
            else:
                s_matrix = torch.matmul(s_matrix, layer.getSMatrix())
        return s_matrix


class FldzhyanModel(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_layers: int,
                 n_classes: int,
                 n_networks: int = 2,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None
                 ):
        super(FldzhyanModel, self).__init__()
        self._n_inputs = n_inputs
        self._n_layers = n_layers
        self._n_classes = n_classes
        self._n_networks = n_networks

        self._photodetectors = Photodetector()

        self._ph_networks = nn.ModuleList()
        self._diag_layer = nn.ParameterList()
        self._biases = nn.ParameterList()

        for i in range(self._n_networks):
            i_losses = None
            imbalance = None
            if mmi_i_losses is not None:
                i_losses = mmi_i_losses[i]
            if mmi_imbalances is not None:
                imbalance = mmi_imbalances[i]

            self._ph_networks.append(FldzhyanNetwork(
                n_layers=self._n_layers,
                n_inputs=self._n_inputs,
                mmi_i_losses=i_losses,
                mmi_imbalances=imbalance
            ))

            self._diag_layer.append(nn.Parameter(
                torch.rand(self._n_inputs), requires_grad=True))

            self._biases.append(nn.Parameter(
                torch.rand(self._n_inputs), requires_grad=True))

    def forward(self, x):

        # Add features with zeros if the architecture N > inputs
        x_reals = self._resizeX(x)

        for i_ph in range(self._n_networks):
            x_complex = torch.complex(x_reals, torch.zeros_like(x_reals))
            y_complex = self._ph_networks[i_ph](x_complex)
            d = self._photodetectors(y_complex)
            x_reals = d * self._diag_layer[i_ph] + self._biases[i_ph]

        lower_limit, upper_limit = self._getLimits()
        y = x_reals[:, lower_limit:upper_limit]
        return y

    def _resizeX(self, x):
        # Check if the number of features exceeds the capacity of the photonic
        # network
        x_size = x.shape[-1]
        if x_size > self._n_inputs:
            raise ValueError(
                f'The number of features exceeds the capacity of the photonic'
                f'network. The maximum allowed number is: {self._n_inputs}'
            )

        # Resize the input data if the number of features is less than the
        # number of inputs of the photnic network
        if x_size < self._n_inputs:
            x_resized = torch.zeros(
                (x.shape[0], self._n_inputs), dtype=x.dtype, device=x.device)

            start_limit = (self._n_inputs - x_size) // 2
            end_limit = start_limit + x_size
            x_resized[..., start_limit:end_limit] = x
            return x_resized
        return x

    def _getLimits(self):
        lower_limit = self._n_inputs//2 - (self._n_classes//2)
        upper_limit = self._n_inputs//2 + self._n_classes//2

        if self._n_classes % 2 != 0:
            upper_limit = upper_limit + 1

        return lower_limit, upper_limit
