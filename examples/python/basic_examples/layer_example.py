from luxia.photonic_meshes import PhotonicLayer, ClementsBlock
from luxia.params import ClementsBlockParams

my_block = ClementsBlock(ClementsBlockParams(number_of_blocks=4))

my_layer = PhotonicLayer(
    n_inputs=8, photonic_block=my_block,
    is_even_layer=True, opt_ps=False
)
