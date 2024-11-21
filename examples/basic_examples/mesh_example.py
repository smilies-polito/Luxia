from photonnn.photonic_meshes import PhotonicMesh, FldzhyanLayer

list_layers = []
for i in range(8):
    is_e = (i % 2) == 0
    layer = FldzhyanLayer(n_inputs=8, is_even_layer=is_e)
    list_layers.append(layer)

my_mesh = PhotonicMesh(n_layers=8, n_inputs=8, layer_list=list_layers)
