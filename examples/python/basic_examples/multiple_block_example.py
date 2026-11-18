import torch
from luxia.photonic_components.beamsplitters import Mmi
from luxia.photonic_components import Heater
from luxia.photonic_meshes import PhotonicBlock
from luxia.params import HeaterParams, MmiParams


# Create a custom block that uses the Heater and Mmi PhotonicComponents
class MyBlock(PhotonicBlock):
    def __init__(self, numOfBlocks: int):
        super(MyBlock, self).__init__(n_blocks=numOfBlocks)
        self._heater = Heater(HeaterParams(number_of_heaters=numOfBlocks))
        self._mmi = Mmi(MmiParams(number_of_splitters=numOfBlocks))

    def forward(self, portsA, portsB):
        outPs = self._heater(portsA)
        outMmi = self._mmi(outPs, portsB)
        return outMmi


# Create 2 random complex signals of size 1 x numOfBlocks
numOfBlocks = 4
Signal_A = torch.complex(torch.rand(1), torch.rand(numOfBlocks))
Signal_B = torch.complex(torch.rand(1), torch.rand(numOfBlocks))

print(Signal_A)
print(Signal_B)
# Instantiate our custom block
block = MyBlock(numOfBlocks=numOfBlocks)

# Call the forward method of the block
output = block(Signal_A, Signal_B)
# print the outputs of the mmi
print(output[0])
print(output[1])
