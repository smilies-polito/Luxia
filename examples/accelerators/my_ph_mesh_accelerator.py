import luxia as lx
import luxia.io_components as io
import costume_mesh.my_ph_mesh as my_msh


class MyPhMeshAccelerator(lx.LxAccelerator):
    def __init__(self,
                 n_inputs: int,
                 n_layers: int,
                 n_classes: int,
                 n_networks: int = 2,
                 mmi_i_losses: [] = None,
                 mmi_imbalances: [] = None,
                 is_matmul: bool = False,
                 ):
        lx_list = []
        for i in range(n_networks):
            i_losses = None if mmi_i_losses is None else mmi_i_losses[i]
            imbalance = None if mmi_imbalances is None else mmi_imbalances[i]

            # create the components of the model and append them to the list
            electrical_to_optical = io.ElectricalToOptical()
            ph_mesh = my_msh.MyCostumeMesh(
                n_layers=n_layers,
                n_inputs=n_inputs,
                mmi_i_losses=i_losses,
                mmi_imbalances=imbalance,
                is_matmul=is_matmul
            )
            optical_to_electrical = lx_list.append(io.OE_PDB(n_inputs))

            lx_list.append(electrical_to_optical)
            lx_list.append(ph_mesh)
            lx_list.append(optical_to_electrical)

        super(MyPhMeshAccelerator, self).__init__(
            n_inputs=n_inputs,
            n_classes=n_classes,
            lx_module_list=lx_list
        )
