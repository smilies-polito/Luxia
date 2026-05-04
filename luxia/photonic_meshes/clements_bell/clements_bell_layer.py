from typing import Optional

from luxia.params import ClementsBellBlockParams
from luxia.photonic_meshes.mesh.photonic_layer import PhotonicLayer

from .clements_bell_block import ClementsBellBlock
from .clements_dual_heater_block import ClementsDualHeaterBlock
from luxia.params import ClementsBellLayerParams


class ClementsBellLayer(PhotonicLayer):
    """One layer of the Clements-Bell mesh."""

    def __init__(
        self,
        cl_params: ClementsBellLayerParams,
    ):
        if cl_params.is_heater_layer:
            hp = None
            if isinstance(cl_params.heater_params, list):
                if len(cl_params.heater_params) > 0:
                    hp = cl_params.heater_params[0]
            else:
                hp = cl_params.heater_params

            photonic_block = ClementsDualHeaterBlock(
                n_blocks=cl_params.n_inputs // 2,
                heater_params=hp,
            )
        else:
            n_blks = cl_params.n_inputs // 2
            if (cl_params.n_inputs % 2 == 0) and (not cl_params.is_even_layer):
                n_blks -= 1

            cb_params = ClementsBellBlockParams(
                n_blocks=n_blks,
                mmi_params=cl_params.mmi_params,
                heater_params=cl_params.heater_params,
            )

            photonic_block = ClementsBellBlock(cb_params=cb_params)

        super(ClementsBellLayer, self).__init__(
            n_inputs=cl_params.n_inputs,
            is_even_layer=cl_params.is_even_layer,
            opt_ps=True,
            photonic_block=photonic_block,
        )
