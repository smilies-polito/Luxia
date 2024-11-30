import luxia as lx
from luxia import photonic_meshes as msh
from luxia import io_components as io


class FldzhyanBellAccelerator(lx.LxAccelerator):
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
            lx_list.append(io.ElectricalToOptical())
            ph_mesh = msh.FldzhyanBellArchitecture(
                n_layers=n_layers,
                n_inputs=n_inputs,
                mmi_i_losses=i_losses,
                mmi_imbalances=imbalance,
                is_matmul=is_matmul
            )
            lx_list.append(ph_mesh)
            lx_list.append(io.OE_PDB(n_inputs))

        super(FldzhyanBellAccelerator, self).__init__(
            n_inputs=n_inputs,
            n_classes=n_classes,
            n_networks=n_networks,
            lx_module_list=lx_list
        )
