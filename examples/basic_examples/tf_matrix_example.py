import torch
from luxia.photonic_components.beamsplitters import Mmi
from luxia.photonic_components import Heater
from luxia.photonic_meshes import PhotonicBlock
from luxia.util.util_functions import diag_matrix_alternating_tensor_with_ones


# Create a custom block that uses the Heater and Mmi PhotonicComponents
class MyBlock(PhotonicBlock):
    def __init__(self, numOfBlocks: int):
        super(MyBlock, self).__init__(n_blocks=numOfBlocks)
        self._heater = Heater(n_heaters=numOfBlocks)
        self._mmi = Mmi(n_mmis=numOfBlocks)

    def get_tf_matrix(self):
        # Get the transfer function matrix of the mmis
        mmi_diag_matrix = self._mmi.get_tf_matrix()
        # The Heater does not provide a get_tf_matrix method, it most used the
        # getPhases method and you must create the matrix manually
        heater_phases = self._heater.getPhases()
        # Create a diagonal matrix with the phases of the heaters
        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(heater_phases)
        # Create the transfer function matrix of the block by multiplying the 
        # matrices of the heaters and the mmis
        tf_matrix = torch.matmul(heater_diag_matrix, mmi_diag_matrix)
        return tf_matrix


# Create 2 random complex signals of size 1 x numOfBlocks
numOfBlocks = 2
# Instantiate our custom block
block = MyBlock(numOfBlocks=numOfBlocks)

my_tf_matrix = block.get_tf_matrix()
print(my_tf_matrix)
