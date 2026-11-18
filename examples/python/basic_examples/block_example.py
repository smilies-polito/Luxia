import torch
from luxia.photonic_components.beamsplitters import Mmi
from luxia.photonic_components import Heater
from luxia.photonic_meshes import PhotonicBlock
from luxia.params import MmiParams


# Create a custom block that uses the Heater and Mmi PhotonicComponents
class MyBlock(PhotonicBlock):
    def __init__(self):
        super(MyBlock, self).__init__()
        self._heater = Heater()
        self._mmi = Mmi(MmiParams())

    def forward(self, portsA, portsB):
        outPs = self._heater(portsA)
        outMmi = self._mmi(outPs, portsB)
        return outMmi


# Create 2 random complex signals of size 1
Signal_A = torch.complex(torch.rand(1), torch.rand(1))
Signal_B = torch.complex(torch.rand(1), torch.rand(1))

# Instantiate our custom block
block = MyBlock()

# Call the forward method of the block
output = block(Signal_A, Signal_B)
# print the outputs of the mmi
print(output[0])
print(output[1])
