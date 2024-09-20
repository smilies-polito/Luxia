import photonnn as pnn
import photonnn.io_components as io
import costume_mesh.my_ph_mesh as ph_arch


class MyPhMeshAccelerator(pnn.PnnAccelerator):
    def __init__(self,
                 n_inputs: int,
                 n_layers: int,
                 n_classes: int,
                 n_networks: int = 2,
                 mmi_i_losses: [] = None,
                 mmi_imbalances: [] = None,
                 is_matmul: bool = False,
                 ):
        pnn_list = []
        for i in range(n_networks):
            i_losses = None if mmi_i_losses is None else mmi_i_losses[i]
            imbalance = None if mmi_imbalances is None else mmi_imbalances[i]

            # create the components of the model and append them to the list
            electrical_to_optical = io.ElectricalToOptical()
            ph_mesh = ph_arch.MyCostumeMesh(
                n_layers=n_layers,
                n_inputs=n_inputs,
                mmi_i_losses=i_losses,
                mmi_imbalances=imbalance,
                is_matmul=is_matmul
            )
            optical_to_electrical = io.OpticalToElectrical(n_inputs)

            pnn_list.append(electrical_to_optical)
            pnn_list.append(ph_mesh)
            pnn_list.append(optical_to_electrical)

        super(MyPhMeshAccelerator, self).__init__(
            n_inputs=n_inputs,
            n_layers=n_layers,
            n_classes=n_classes,
            n_networks=n_networks,
            pnn_module_list=pnn_list
        )
