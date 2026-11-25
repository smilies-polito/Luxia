from typing import Any, Dict, List, Tuple

import torch

from luxia.photonic_meshes.mesh import PhotonicMesh
from luxia.photonic_meshes.clements.clements_layer import ClementsLayer
from luxia.params import (
    ClementsLayerParams,
    ClementsMeshParams,
)


class ClementsMesh(PhotonicMesh):
    """
    Clements rectangular photonic mesh built from bound C++ ClementsLayer objects.

    This mesh follows the rectangular interferometer topology introduced in:

        William R. Clements, Peter C. Humphreys, Benjamin J. Metcalf,
        W. Steven Kolthammer, Ian A. Walmsley,
        "An Optimal Design for Universal Multiport Interferometers",
        Optica 3(12), 1460-1465 (2016).
        DOI: 10.1364/OPTICA.3.001460

    Reference:
        https://doi.org/10.1364/OPTICA.3.001460

    The mesh is defined as an ordered sequence of ClementsLayerParams objects.
    The user must provide exactly n_layers layer-parameter objects, one per
    physical layer in the rectangular mesh topology.

    The Clements rectangular arrangement is an alternative beam-splitter and
    phase-shifter mesh for universal multiport interferometers; the paper
    reports half the optical depth of the Reck design and improved robustness
    to optical loss. [web:62]
    """

    def __init__(
        self,
        params: ClementsMeshParams,
        debug: bool = False,
    ) -> None:
        """
        Initialize the Clements mesh from explicit per-layer parameters.
        """
        self._total_i_losses: torch.Tensor = torch.tensor(0.0)

        layer_list: List[ClementsLayer] = []
        self._validate_mesh_params(params)

        for layer_index, layer_params in enumerate(params.layer_params):
            self._validate_layer_position(
                mesh_params=params,
                layer_index=layer_index,
                layer_params=layer_params,
            )
            layer_list.append(ClementsLayer(params=layer_params, debug=debug))

        super().__init__(
            n_layers=params.number_of_layers,
            n_inputs=params.number_of_inputs,
            is_matmul=params.is_matmul,
            layer_list=layer_list,
            debug=debug,
        )

    # ---------------------------------------------------------------------
    # Validation helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _validate_mesh_params(mesh_params: ClementsMeshParams) -> None:
        """
        Validate mesh-level structural consistency.
        """
        if mesh_params.number_of_inputs <= 0:
            raise ValueError("number_of_inputs must be > 0")

        if mesh_params.number_of_layers <= 0:
            raise ValueError("number_of_layers must be > 0")

        if len(mesh_params.layer_params) != mesh_params.number_of_layers:
            raise ValueError(
                "layer_params must have the same length as number_of_layers"
            )

    @staticmethod
    def _validate_layer_position(
        mesh_params: ClementsMeshParams,
        layer_index: int,
        layer_params: ClementsLayerParams,
    ) -> None:
        """
        Validate that one layer parameter object matches its mesh position.
        """
        if layer_params.number_of_inputs != mesh_params.number_of_inputs:
            raise ValueError(
                f"layer_params[{layer_index}].number_of_inputs must equal "
                f"mesh number_of_inputs ({mesh_params.number_of_inputs}), "
                f"got {layer_params.number_of_inputs}"
            )

        expected_even_layer = (layer_index % 2) == 0
        if layer_params.is_even_layer != expected_even_layer:
            raise ValueError(
                f"layer_params[{layer_index}].is_even_layer must be "
                f"{expected_even_layer} for layer index {layer_index}"
            )

    # ---------------------------------------------------------------------
    # Parameter re-application / export
    # ---------------------------------------------------------------------
    def set_params(self, params: ClementsMeshParams) -> None:
        """
        Re-apply mesh parameters to the existing layers.
        """
        self._validate_mesh_params(params)

        if params.number_of_inputs != self.get_n_inputs():
            raise ValueError(
                "ClementsMesh.set_params: number_of_inputs shall not change"
            )

        if params.number_of_layers != self.get_n_layers():
            raise ValueError(
                "ClementsMesh.set_params: number_of_layers shall not change"
            )

        if params.is_matmul != self.get_is_matmul():
            raise ValueError(
                "ClementsMesh.set_params: is_matmul shall not change"
            )

        for layer_index, (layer, layer_params) in enumerate(
            zip(self.get_photonic_layers(), params.layer_params)
        ):
            self._validate_layer_position(
                mesh_params=params,
                layer_index=layer_index,
                layer_params=layer_params,
            )
            layer.set_params(params=layer_params)

    def get_params(self) -> ClementsMeshParams:
        """
        Export the current mesh parameterization.
        """
        layer_params: List[ClementsLayerParams] = []

        for layer in self.get_photonic_layers():
            layer_params.append(layer.get_params())

        return ClementsMeshParams(
            number_of_inputs=self.get_n_inputs(),
            number_of_layers=self.get_n_layers(),
            layer_params=layer_params,
            is_matmul=self.get_is_matmul(),
        )

    # ---------------------------------------------------------------------
    # Weight / phase helpers
    # ---------------------------------------------------------------------
    def get_weights_tensor(self) -> torch.Tensor:
        """
        Return all heater phases as one flattened tensor.
        """
        phase_list = self.get_layer_phase_shifts()

        if not phase_list:
            return torch.empty(0)

        flat_phase_list = [phase.reshape(-1) for phase in phase_list]
        return torch.cat(flat_phase_list, dim=0)

    # ---------------------------------------------------------------------
    # Metrics helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _to_scalar_tensor(value: Any) -> torch.Tensor:
        """
        Convert a metric value to a scalar float32 tensor.
        """
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.reshape(())
            return value.to(dtype=torch.float32).mean().reshape(())

        return torch.tensor(float(value), dtype=torch.float32)

    def _get_layer_heater_metrics_payload(
        self,
        layer: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Return the raw heater metrics payload for one layer.
        """
        metrics = layer.get_heater_metrics()
        heater_metrics = metrics.get("heater_metrics", [])
        return heater_metrics, metrics

    def get_total_insertion_losses(self) -> torch.Tensor:
        """
        Return the total insertion loss of the mesh in dB.
        """
        input_power = self.get_input_optical_power()
        output_power = self.get_output_optical_power()
        self._total_i_losses = 10.0 * torch.log10(input_power / output_power)
        return self._total_i_losses

    def get_heaters_quantization_errors(self) -> List[List[List[torch.Tensor]]]:
        """
        Return heater quantization errors for all layers.

        The return structure is aligned with the existing mesh/layer/block
        traversal used by downstream aggregation code.
        """
        mesh_errors: List[List[List[torch.Tensor]]] = []

        for layer in self.get_photonic_layers():
            heater_metrics, _ = self._get_layer_heater_metrics_payload(layer)

            heater_errors = [
                self._to_scalar_tensor(metric.get("quantization_error", 0.0))
                for metric in heater_metrics
            ]

            mesh_errors.append([heater_errors])

        return mesh_errors

    def get_heaters_snr(self) -> torch.Tensor:
        """
        Return the average heater SNR across layers.
        """
        snr_per_layer: List[torch.Tensor] = []

        for layer in self.get_photonic_layers():
            heater_metrics, _ = self._get_layer_heater_metrics_payload(layer)
            snr_values = [
                float(metric.get("snr_db", 0.0))
                for metric in heater_metrics
            ]

            if snr_values:
                snr_per_layer.append(
                    torch.tensor(snr_values, dtype=torch.float32)
                )

        if not snr_per_layer:
            return torch.tensor(0.0, dtype=torch.float32)

        return torch.stack(snr_per_layer, dim=0).mean(dim=0)

    def get_heaters_sqnr(self) -> torch.Tensor:
        """
        Return the average heater SQNR across layers.
        """
        sqnr_per_layer: List[torch.Tensor] = []

        for layer in self.get_photonic_layers():
            heater_metrics, _ = self._get_layer_heater_metrics_payload(layer)
            sqnr_values = [
                float(metric.get("sqnr_db", 0.0))
                for metric in heater_metrics
            ]

            if sqnr_values:
                sqnr_per_layer.append(
                    torch.tensor(sqnr_values, dtype=torch.float32)
                )

        if not sqnr_per_layer:
            return torch.tensor(0.0, dtype=torch.float32)

        return torch.stack(sqnr_per_layer, dim=0).mean(dim=0)

    def get_power_consumption(self) -> torch.Tensor:
        """
        Return the total heater power consumption of the mesh.
        """
        total_power = 0.0

        for layer in self.get_photonic_layers():
            _, metrics = self._get_layer_heater_metrics_payload(layer)
            total_power += float(metrics.get("total_power_consumption", 0.0))

        return torch.tensor(total_power, dtype=torch.float32)

    # ---------------------------------------------------------------------
    # Canonical metrics API
    # ---------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """
        Return mesh metrics in dictionary form.

        This is the canonical public metrics API for module-level
        metric aggregation.
        """
        return {
            "total_i_losses": self.get_total_insertion_losses(),
            "heaters_snr": self.get_heaters_snr(),
            "heaters_sqnr": self.get_heaters_sqnr(),
            "power_consumption": self.get_power_consumption(),
            "heaters_quantization_errors": (
                self.get_heaters_quantization_errors()
            ),
        }
