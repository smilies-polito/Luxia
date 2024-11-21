import photonnn.photonic_meshes as ph_arch

# Create a photonic mesh
ph_mesh = ph_arch.FldzhyanMesh(n_inputs=4, n_layers=4, is_matmul=False)

# Create a random input tensor
transfer_matrix = ph_mesh.get_tf_matrix()
print(transfer_matrix)

# Get the transfer matrix of eah layer of the mesh
for layer in ph_mesh._ph_layers:
    transfer_matrix = layer.get_tf_matrix()
    print(transfer_matrix)
