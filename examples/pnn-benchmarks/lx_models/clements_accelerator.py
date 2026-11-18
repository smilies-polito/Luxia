import luxia as lx
from luxia import photonic_meshes as msh
from luxia import io_components as io
from luxia.params.clements_mesh_params import ClementsMeshParams


class ClementsAccelerator(lx.LxAccelerator):
    def __init__(self,
                 n_inputs: int,
                 n_layers: int,
                 n_classes: int,
                 n_networks: int = 2,
                 mmi_i_losses: [] = None,
                 mmi_imbalances: [] = None,
                 is_matmul: bool = False,
                 window_size: int = 1,
                 ):
        lx_list = []

        p = ClementsMeshParams(
            number_of_inputs=n_inputs,
            number_of_layers=n_layers,
            is_matmul=is_matmul,
            window_size=window_size
        )

        for i in range(n_networks):
            lx_list.append(io.ElectricalToOptical())

            ph_mesh = msh.ClementsMesh(clements_mesh_params=p)
            lx_list.append(ph_mesh)

            lx_list.append(io.OpticalToElectrical())

        super(ClementsAccelerator, self).__init__(
            n_inputs=n_inputs,
            n_classes=n_classes,
            lx_module_list=lx_list
        )
