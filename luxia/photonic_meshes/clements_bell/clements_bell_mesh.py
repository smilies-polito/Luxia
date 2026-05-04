from luxia.params import (
    ClementsBellMeshParams,
    ClementsBellLayerParams,
)
from luxia.photonic_meshes.mesh.photonic_mesh import PhotonicMesh
from .clements_bell_layer import ClementsBellLayer


class ClementsBellMesh(PhotonicMesh):
    """Clements-Bell photonic mesh."""

    def __init__(self, cm_params: ClementsBellMeshParams):
        layer_list = []

        first_layer_params = ClementsBellLayerParams(
            n_inputs=cm_params.n_inputs,
            is_even_layer=True,
            heater_params=cm_params.heater_params,
            is_heater_layer=True,
        )
        layer_list.append(ClementsBellLayer(cl_params=first_layer_params))

        for i in range(cm_params.n_layers - 2):
            p = None if cm_params.mmis_params is None else cm_params.mmis_params[i]

            layer_params = ClementsBellLayerParams(
                n_inputs=cm_params.n_inputs,
                is_even_layer=(i % 2 == 0),
                mmi_params=p,
                heater_params=cm_params.heater_params,
                is_heater_layer=False,
            )
            layer_list.append(ClementsBellLayer(cl_params=layer_params))

        last_layer_params = ClementsBellLayerParams(
            n_inputs=cm_params.n_inputs,
            is_even_layer=True,
            heater_params=cm_params.heater_params,
            is_heater_layer=True,
        )
        layer_list.append(ClementsBellLayer(cl_params=last_layer_params))

        super(ClementsBellMesh, self).__init__(
            n_layers=cm_params.n_layers,
            n_inputs=cm_params.n_inputs,
            is_matmul=cm_params.is_matmul,
            layer_list=layer_list,
            window_size=cm_params.window_size,
        )
