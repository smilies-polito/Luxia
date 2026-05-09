import luxia as lx
from luxia import photonic_meshes as ph_arch
from luxia import io_components as io
from luxia.params import ClementsBellMeshParams


class ClementsBellAccelerator(lx.LxAccelerator):
    def __init__(
        self,
        n_inputs: int,
        n_layers: int,
        n_classes: int,
        n_networks: int = 2,
        mmi_params=None,
        heater_params=None,
        is_matmul: bool = False,
        window_size: int = 1
    ):
        lx_list = []

        for _ in range(n_networks):
            lx_list.append(io.ElectricalToOptical())

            cm_params = ClementsBellMeshParams(
                n_layers=n_layers,
                n_inputs=n_inputs,
                mmis_params=mmi_params,
                heater_params=heater_params,
                is_matmul=is_matmul,
                window_size=window_size
            )

            ph_mesh = ph_arch.ClementsBellMesh(cm_params=cm_params)
            lx_list.append(ph_mesh)
            lx_list.append(io.OE_PDB(n_inputs=n_inputs))

        super(ClementsBellAccelerator, self).__init__(
            n_inputs=n_inputs,
            n_classes=n_classes,
            lx_module_list=lx_list,
        )
