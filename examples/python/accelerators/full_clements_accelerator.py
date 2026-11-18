from luxia.photonic_components import IdealPhotodetector, LaserModulator
from luxia.params import (
    DacParams, AdcParams, HeaterParams,
    ClementsMeshParams, ClementsLayerParams, ClementsBlockParams,
)
from typing import List, Optional, Literal
from luxia import photonic_meshes as msh
from luxia import io_components as io
from collections import OrderedDict
from luxia.types import round_t
import luxia as lx
import torch


class FullClementsAccelerator(lx.LxAccelerator):
    """
    A full Clements mesh accelerator for photonic neural networks.

    This class implements a photonic accelerator using Clements mesh
    architecture, including DACs, laser modulators, photonic meshes,
    photodetectors, and ADCs.
    """

    def __init__(
        self,
        n_inputs: int,
        n_layers: int,
        n_classes: int,
        n_networks: int = 2,
        is_matmul: bool = False,
        dac_resolution: Optional[List[int]] = None,
        adc_resolution: Optional[List[int]] = None,
        heaters_resolution: Optional[List[int]] = None,
        rndi_type: Literal["ceil", "floor", "round"] = "ceil",
    ):
        """
        Initialize the FullClementsAccelerator.

        Args:
            n_inputs: Number of input features
            n_layers: Number of layers in the photonic mesh
            n_classes: Number of output classes
            n_networks: Number of parallel photonic networks
            is_matmul: Whether to use matrix multiplication mode
            dac_resolution: List of bits for input DAC resolution
            adc_resolution: List of bits for output ADC resolution
            heaters_resolution: List of bits for heater DAC resolution
            rndi_type: Rounding type ("ceil", "floor", or "round")
        """
        # Initialize instance variables
        self._n_networks = n_networks

        # Create an OrderedDict to store all photonic components
        lx_list = OrderedDict()

        # Get rounding type
        r_type = self._get_rounding_type_from_str(rndi_type)

        # Input DAC parameters
        dac_params = DacParams(
            number_of_bits=dac_resolution or 8,
            min_digital_reference=0.0,
            max_digital_reference=1.0,
            min_analog_reference=0.0,
            max_analog_reference=1.0,
            round_type=r_type,
        )
        lx_list["DAC"] = io.Dac(dac_params, debug=False)

        # Heater DAC parameters (optional per-heater quantisation)
        heater_dac = DacParams(
            number_of_bits=heaters_resolution or 8,
            min_digital_reference=0.0,
            max_digital_reference=float(2.0 * torch.pi),
            min_analog_reference=0.0,
            max_analog_reference=float(2.0 * torch.pi),
            round_type=r_type,
        )
        heater_params = HeaterParams(
            dac_params=heater_dac,
            use_dac=(heaters_resolution is not None),
        )
        block_params = ClementsBlockParams(
            number_of_blocks=n_inputs // 2,
            heater_params=[heater_params, heater_params],
        )

        # Create Photonic Components for each network
        for i in range(n_networks):

            lx_list[f"LaserModulator_{i}"] = LaserModulator(lmod_params=None)

            # Build Clements mesh with optional heater DAC resolution
            layer_params = [
                ClementsLayerParams(
                    number_of_inputs=n_inputs,
                    is_even_layer=(j % 2 == 0),
                    clements_blk_params=block_params,
                )
                for j in range(n_layers)
            ]
            mesh_params = ClementsMeshParams(
                number_of_inputs=n_inputs, number_of_layers=n_layers,
                layer_params=layer_params, is_matmul=is_matmul,
            )
            mesh = msh.ClementsMesh(params=mesh_params)
            lx_list[f"PhotonicMesh_{i}"] = mesh

            # Create Photodetector
            lx_list[f"Photodetector_{i}"] = IdealPhotodetector(debug=True)

        # ADC Initialization
        adc_params = AdcParams(
            number_of_bits=adc_resolution or 8,
            max_analog_reference=float(torch.sqrt(torch.tensor(float(n_inputs)))),
            min_analog_reference=0.0,
            round_type=r_type,
        )

        lx_list["ADC"] = io.Adc(adc_params, debug=False)

        # Call the parent class constructor
        super().__init__(
            n_inputs=n_inputs,
            n_classes=n_classes,
            lx_module_list=lx_list
        )

    def _configure_mesh_dacs(self, mesh, debug: bool = False) -> None:
        """Configure all DACs in the mesh with the specified debug mode.
        Args:
            mesh: The photonic mesh containing DACs
            debug: Whether to set DACs in debug mode
        """
        for layer in mesh.get_photonic_layers():
            blk = layer.get_photonic_block()
            blk._heater._dac.set_debug_mode(debug)
            blk._mzi._heater._dac.set_debug_mode(debug)

    @staticmethod
    def _get_rounding_type_from_str(round_str: str) -> round_t:
        """Convert string rounding type to round_t enum.
        Args:
            round_str: String representation of rounding type
        Returns:
            Corresponding round_t enum value
        Raises:
            ValueError: If round_str is not a valid rounding type
        """
        round_types = {
            "ceil": round_t.CEIL,
            "floor": round_t.FLOOR,
            "round": round_t.ROUND
        }

        if round_str not in round_types:
            raise ValueError(
                f"Invalid round type: {round_str}."
                f" Must be one of: {', '.join(round_types.keys())}")

        return round_types[round_str]

    def get_dac(self) -> io.Dac:
        """Get the Digital-to-Analog Converter.
        Returns:
            The DAC component
        """
        return self._ph_model["DAC"]

    def get_adc(self) -> io.Adc:
        """Get the Analog-to-Digital Converter.
        Returns:
            The ADC component
        """
        return self._ph_model["ADC"]

    def get_photonic_meshes(self) -> List[msh.ClementsMesh]:
        """Retrieve all photonic meshes.
        Returns:
            List of photonic mesh components
        """
        meshes = []
        for i in range(self._n_networks):
            meshes.append(self._ph_model[f"PhotonicMesh_{i}"])
        return meshes

    def get_laser_modulators(self) -> List[LaserModulator]:
        """Retrieve all laser modulators.
        Returns:
            List of laser modulator components
        """
        lsm = []
        for i in range(self._n_networks):
            lsm.append(self._ph_model[f"LaserModulator_{i}"])
        return lsm

    def get_photodetectors(self) -> List[IdealPhotodetector]:
        """Retrieve all photodetectors.
        Returns:
            List of photodetector components
        """
        pds = []
        for i in range(self._n_networks):
            pds.append(self._ph_model[f"Photodetector_{i}"])
        return pds

    # Heaters methods
    def get_heaters_sqnr(self) -> List[torch.Tensor]:
        """Get signal-to-quantization-noise ratio for heaters in each mesh.
        Returns:
            List of SQNR values for heaters in each mesh
        """
        return [mesh.get_heaters_sqnr() for mesh in self.get_photonic_meshes()]

    def get_heaters_snr(self) -> List[torch.Tensor]:
        """Get signal-to-noise ratio for heaters in each mesh.
        Returns:
            List of SNR values for heaters in each mesh
        """
        return [mesh.get_heaters_snr() for mesh in self.get_photonic_meshes()]

    def get_heaters_avg_snr(self) -> torch.Tensor:
        """Get average signal-to-noise ratio across all meshes.
        Returns:
            Average SNR tensor
        """
        heater_snr = self.get_heaters_snr()
        return torch.stack(heater_snr, dim=0).mean(dim=0)

    def get_heaters_quantization_error(self) -> List[torch.Tensor]:
        """Get quantization error for heaters in each photonic mesh.
        Returns:
            List of quantization errors for heaters in each mesh
        """
        qe = []
        for mesh in self.get_photonic_meshes():
            mesh_qe = mesh.get_heaters_quantization_error()
            qe.append(mesh_qe)
        return qe

    def get_heaters_average_qerror(self) -> float:
        """Get the average quantization error for all heaters in all meshes.
        Returns:
            Average quantization error as a float
        """
        all_errors = []

        for mesh in self.get_photonic_meshes():
            mesh_errors = mesh.get_heaters_quantization_error()
            for layer_errors in mesh_errors:
                for heater_errors in layer_errors:
                    all_errors.extend([qe.item() for qe in heater_errors])
        return sum(all_errors) / len(all_errors) if all_errors else 0.0

    # DAC Methods
    def get_dac_sqnr(self) -> torch.Tensor:
        """Get signal-to-quantization-noise ratio for the DAC.
        Returns:
            SQNR tensor for the DAC
        """
        return self.get_dac().get_sqnr()

    def get_dac_snr(self) -> torch.Tensor:
        """Get signal-to-noise ratio for the DAC.
        Returns:
            SNR tensor for the DAC
        """
        return self.get_dac().get_snr()

    def get_dac_quantization_error(self) -> torch.Tensor:
        """Get quantization error for the DAC.
        Returns:
            Quantization error tensor for the DAC
        """
        return self.get_dac().get_quantization_error()

    def get_dac_average_qerror(self) -> float:
        """Get the average quantization error for the DAC.
        Returns:
            Average quantization error as a float
        """
        return torch.mean(self.get_dac_quantization_error()).item()

    # ADC Methods
    def get_adc_sqnr(self) -> torch.Tensor:
        """Get signal-to-quantization-noise ratio for the ADC.
        Returns:
            SQNR tensor for the ADC
        """
        return self.get_adc().get_sqnr()

    def get_adc_snr(self) -> torch.Tensor:
        """Get signal-to-noise ratio for the ADC.
        Returns:
            SNR tensor for the ADC
        """
        return self.get_adc().get_snr()

    def get_adc_quantization_error(self) -> torch.Tensor:
        """Get quantization error for the ADC.
        Returns:
            Quantization error tensor for the ADC
        """
        return self.get_adc().get_quantization_error()

    def get_adc_average_qerror(self) -> float:
        """Get the average quantization error for the ADC.
        Returns:
            Average quantization error as a float
        """
        return torch.mean(self.get_adc_quantization_error()).item()
