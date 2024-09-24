from photonnn.photonic_architectures import PhotonicLayer, ClementsBlock

my_block = ClementsBlock(n_blocks=4)

my_layer = PhotonicLayer(
    n_inputs=8, n_blocks=4, photonic_block=my_block,
    even_layer=True, opt_ps=False
)
