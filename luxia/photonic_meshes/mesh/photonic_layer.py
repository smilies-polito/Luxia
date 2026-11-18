from luxia.photonic_meshes.mesh import PhotonicBlock
from luxia.photonic_components import Heater
from luxia.util.util_functions import alternating_tensors
from luxia.util.log_config import lx_log
from luxia import LxModule
from torch import nn
import torch


class PhotonicLayer(LxModule):
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
                 photonic_block: PhotonicBlock,
                 is_even_layer: bool = True,
                 opt_ps: bool = False,
                 debug: bool = False):
        super().__init__(debug=debug)
        self._n_inputs = n_inputs
        self._is_even_layer = is_even_layer
        self._opt_ps = opt_ps
        self._is_even_input = (n_inputs % 2 == 0)
        
        # Single source of truth for forward method selection
        self._forward_method = self._select_forward_method()
        
        # Heaters only when needed
        self._heater = self._init_heaters()
        self._ph_blocks = photonic_block

    def _select_forward_method(self):
        """Single method to select forward based on input/layer parity"""
        if self._is_even_input:
            return self._forward_even_even if self._is_even_layer else self._forward_even_odd
        return self._forward_odd_even if self._is_even_layer else self._forward_odd_odd

    def _init_heaters(self):
        """Initialize heaters only when opt_ps is True and needed"""
        if not self._opt_ps:
            return None
        if self._is_even_input and not self._is_even_layer:
            return nn.ModuleList([Heater(1), Heater(1)])
        elif not self._is_even_input:
            return Heater(1)
        return None

    def forward(self, x_complex):
        """Unified entry point - delegates to selected method"""
        return self._forward_method(x_complex)

    def _debug_log_tensors(self, x_complex, y_complex):
        """Unified debug logging matching PhotonicMesh style"""
        if not self._debug:
            return
            
        torch.set_printoptions(profile="full")
        lx_log.setLevel("DEBUG")

        # Safely get first vector
        x_first = x_complex[0] if x_complex.dim() >= 2 else x_complex
        y_first = y_complex[0] if y_complex.dim() >= 2 else y_complex
        
        input_power = torch.sum(torch.abs(x_first) ** 2)
        output_power = torch.sum(torch.abs(y_first) ** 2)
        
        lx_log.debug("=== PhotonicLayer debug ===")
        lx_log.debug(f"x (first vec):\n{x_first}")
        lx_log.debug(f"|x| (first vec):\n{torch.abs(x_first)}")
        lx_log.debug(f"|x|^2 (first vec):\n{torch.abs(x_first) ** 2}")
        lx_log.debug(f"sum(|x|^2): {input_power}")
        
        lx_log.debug(f"y (first vec):\n{y_first}")
        lx_log.debug(f"|y| (first vec):\n{torch.abs(y_first)}")
        lx_log.debug(f"|y|^2 (first vec):\n{torch.abs(y_first) ** 2}")
        lx_log.debug(f"sum(|y|^2): {output_power}")
        lx_log.debug(f"Power loss: {input_power - output_power:.2e}")
        
        input("Press Enter to continue...")
        torch.set_printoptions(profile="default")

    def _forward_even_even(self, x_complex):
        """Even inputs, even layer - Fig 1.A"""
        x_complex_0 = x_complex[:, ::2]  # All A's
        x_complex_1 = x_complex[:, 1::2]  # All B's
        y_0, y_1 = self._ph_blocks(x_complex_0, x_complex_1)
        y_complex = alternating_tensors(y_0, y_1)
        self._debug_log_tensors(x_complex, y_complex)
        return y_complex

    def _forward_even_odd(self, x_complex):
        """Even inputs, odd layer - Fig 1.B"""
        y_complex = torch.zeros_like(x_complex)
        
        # Direct paths A0->C0, AN->CN
        if self._opt_ps:
            y_complex[:, 0] = self._heater[0](x_complex[:, 0])
            y_complex[:, -1] = self._heater[1](x_complex[:, -1])
        else:
            y_complex[:, 0] = x_complex[:, 0]
            y_complex[:, -1] = x_complex[:, -1]
        
        # PHBlocks for middle elements
        x_mid = x_complex[:, 1:-1]
        x_complex_0 = x_mid[:, ::2]  # B's
        x_complex_1 = x_mid[:, 1::2]  # A's
        y_0, y_1 = self._ph_blocks(x_complex_0, x_complex_1)
        y_complex[:, 1:-1] = alternating_tensors(y_0, y_1)
        
        self._debug_log_tensors(x_complex, y_complex)
        return y_complex

    def _forward_odd_even(self, x_complex):
        """Odd inputs, even layer - Fig 2.A"""
        y_complex = torch.zeros_like(x_complex)
        
        # Direct path AN->CN
        if self._opt_ps:
            y_complex[:, -1] = self._heater(x_complex[:, -1])
        else:
            y_complex[:, -1] = x_complex[:, -1]
        
        # PHBlocks for first N-1 elements
        x_prefix = x_complex[:, :-1]
        x_complex_0 = x_prefix[:, ::2]  # A's
        x_complex_1 = x_prefix[:, 1::2]  # B's
        y_0, y_1 = self._ph_blocks(x_complex_0, x_complex_1)
        y_complex[:, :-1] = alternating_tensors(y_0, y_1)
        
        self._debug_log_tensors(x_complex, y_complex)
        return y_complex

    def _forward_odd_odd(self, x_complex):
        """Odd inputs, odd layer - Fig 2.B"""
        y_complex = torch.zeros_like(x_complex)
        
        # Direct path A0->C0
        if self._opt_ps:
            y_complex[:, 0] = self._heater(x_complex[:, 0])
        else:
            y_complex[:, 0] = x_complex[:, 0]
        
        # PHBlocks for remaining elements
        x_suffix = x_complex[:, 1:]
        x_complex_0 = x_suffix[:, ::2]  # B's
        x_complex_1 = x_suffix[:, 1::2]  # A's
        y_0, y_1 = self._ph_blocks(x_complex_0, x_complex_1)
        y_complex[:, 1:] = alternating_tensors(y_0, y_1)
        
        self._debug_log_tensors(x_complex, y_complex)
        return y_complex

    def get_tf_matrix(self):
        """Transfer matrix construction - simplified and device-aware"""
        device = next(self.parameters()).device if self.parameters() else torch.device('cpu')
        
        if self._is_even_input:
            tf_core = self._ph_blocks.get_tf_matrix()
            if self._is_even_layer:
                return tf_core
            else:
                ones = torch.ones(1, device=device)
                return torch.block_diag(ones, tf_core, ones)
        else:
            tf_core = self._ph_blocks.get_tf_matrix()
            ones = torch.ones(1, device=device)
            return torch.block_diag(tf_core, ones) if self._is_even_layer else torch.block_diag(ones, tf_core)

    # Clean getters
    def get_photonic_block(self): return self._ph_blocks
    def get_is_even_layer(self): return self._is_even_layer
    def get_n_inputs(self): return self._n_inputs
    def get_opt_ps(self): return self._opt_ps
