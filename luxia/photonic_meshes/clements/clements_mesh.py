from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from luxia.photonic_meshes.mesh import PhotonicMesh
from luxia.params import (
    BsMmiParams,
    ClementsBlockParams,
    ClementsLayerParams,
    ClementsMeshParams,
    DacParams,
    HeaterParams,
)
from luxia.types import lx_real  # Add this import
from .clements_layer import ClementsLayer


@dataclass
class MeshMetrics:
    total_i_losses: torch.Tensor
    heaters_snr: torch.Tensor
    heaters_sqnr: torch.Tensor
    power_consumption: torch.Tensor


class ClementsMesh(PhotonicMesh):
    """
    Clements photonic mesh built from bound C++ ClementsLayer objects.

    Each layer is constructed directly with its final per-layer heater/MMI
    parameters so that the underlying C++ modules are born with the correct
    trainable parameter sizes.
    """

    def __init__(
        self,
        clements_mesh_params: ClementsMeshParams,
        debug: bool = False,
    ) -> None:
        self._total_i_losses: torch.Tensor = lx_real(0.0)  # Dtype-aware zero

        list_layers: List[ClementsLayer] = []

        for layer_index in range(clements_mesh_params.n_layers):
            layer_params = self._build_layer_params(
                mesh_params=clements_mesh_params,
                layer_index=layer_index,
            )
            list_layers.append(ClementsLayer(params=layer_params))

        super().__init__(
            n_layers=clements_mesh_params.n_layers,
            n_inputs=clements_mesh_params.n_inputs,
            is_matmul=clements_mesh_params.is_matmul,
            layer_list=list_layers,
            debug=debug,
            window_size=clements_mesh_params.window_size
        )

    # ------------------------------------------------------------------
    # Topology / parameter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_num_blocks(n_inputs: int, layer_index: int) -> int:
        n_blks = n_inputs // 2
        is_even_layer = (layer_index % 2 == 0)
        if (n_inputs % 2 == 0) and not is_even_layer:
            n_blks -= 1
        return n_blks

    @staticmethod
    def _clone_param(obj: Any) -> Any:
        return deepcopy(obj)

    @classmethod
    def _normalize_param_list(
        cls,
        param_list: Optional[List[Any]],
        blk_size: int,
        name: str,
    ):
        if param_list is None:
            return None

        length = len(param_list)
        assert length in (1, 2), (
            f"{name} must have length 1 or 2, got {length}"
        )

        def _count_attr(obj: object) -> str:
            if hasattr(obj, "n_heaters"):
                return "n_heaters"
            if hasattr(obj, "num_splitters"):
                return "num_splitters"
            raise AttributeError(
                f"{name} entries must expose 'n_heaters' or 'num_splitters'"
            )

        if length == 1:
            base = cls._clone_param(param_list[0])
            count_attr = _count_attr(base)
            setattr(base, count_attr, blk_size)

            other = cls._clone_param(base)
            return [base, other]

        out = []
        for idx, obj in enumerate(param_list):
            cloned = cls._clone_param(obj)
            count_attr = _count_attr(cloned)
            count_val = getattr(cloned, count_attr)

            assert count_val in (1, blk_size), (
                f"{name}[{idx}] has {count_attr}={count_val}, "
                f"expected 1 or {blk_size}"
            )

            setattr(cloned, count_attr, blk_size)
            out.append(cloned)

        return out

    @classmethod
    def _select_layer_param_spec(
        cls,
        all_params,
        layer_index: int,
        param_name: str,
    ):
        """
        Select the parameter spec for one layer.

        Supported conventions:
        - None
        - [[left, right]] shared across all layers
        - [[[...], [...], ...]] one entry per layer
        """
        if all_params is None:
            return None

        if len(all_params) == 0:
            return None

        if len(all_params) == 1:
            return all_params[0]

        assert layer_index < len(all_params), (
            f"{param_name} has {len(all_params)} layer entries, "
            f"but requested layer_index={layer_index}"
        )
        return all_params[layer_index]

    @classmethod
    def _resolve_heater_params(
        cls,
        mesh_params: ClementsMeshParams,
        layer_index: int,
        n_blks: int,
    ):
        if mesh_params.heater_params is None:
            h0 = HeaterParams(
                n_heaters=n_blks,
                dac_params=DacParams(),
                use_dac=False,
            )
            h1 = HeaterParams(
                n_heaters=n_blks,
                dac_params=DacParams(),
                use_dac=False,
            )
            return [h0, h1]

        ht_list = cls._select_layer_param_spec(
            all_params=mesh_params.heater_params,
            layer_index=layer_index,
            param_name="heater_params",
        )

        return cls._normalize_param_list(
            param_list=ht_list,
            blk_size=n_blks,
            name="heater_params",
        )

    @classmethod
    def _resolve_mmi_params(
        cls,
        mesh_params: ClementsMeshParams,
        layer_index: int,
        n_blks: int,
    ):
        if mesh_params.mmis_params is None:
            m0 = BsMmiParams(num_splitters=n_blks)
            m1 = BsMmiParams(num_splitters=n_blks)
            return [m0, m1]

        mmi_list = cls._select_layer_param_spec(
            all_params=mesh_params.mmis_params,
            layer_index=layer_index,
            param_name="mmis_params",
        )

        return cls._normalize_param_list(
            param_list=mmi_list,
            blk_size=n_blks,
            name="mmis_params",
        )

    @classmethod
    def _build_layer_params(
        cls,
        mesh_params: ClementsMeshParams,
        layer_index: int,
    ) -> ClementsLayerParams:
        n_blks = cls._compute_num_blocks(
            n_inputs=mesh_params.n_inputs,
            layer_index=layer_index,
        )
        is_even_layer = (layer_index % 2 == 0)

        heater_params = cls._resolve_heater_params(
            mesh_params=mesh_params,
            layer_index=layer_index,
            n_blks=n_blks,
        )
        mmi_params = cls._resolve_mmi_params(
            mesh_params=mesh_params,
            layer_index=layer_index,
            n_blks=n_blks,
        )

        blk_p = ClementsBlockParams(
            n_blocks=n_blks,
            mmi_params=mmi_params,
            heater_params=heater_params,
        )

        return ClementsLayerParams(
            n_inputs=mesh_params.n_inputs,
            is_even_layer=is_even_layer,
            opt_ps=False,
            clements_blk_params=blk_p,
        )

    # ------------------------------------------------------------------
    # Parameter re-application / export
    # ------------------------------------------------------------------

    def set_params(self, params: ClementsMeshParams) -> None:
        """
        Re-apply mesh parameters to all existing layers.

        This is still allowed for explicit reconfiguration after construction,
        but __init__ no longer depends on it.
        """
        for i, layer in enumerate(self.get_photonic_layers()):
            layer_params = self._build_layer_params(
                mesh_params=params,
                layer_index=i,
            )
            layer.set_params(params=layer_params)

    def get_params(self) -> ClementsMeshParams:
        n_layers = self.get_n_layers()
        n_inputs = self.get_n_inputs()

        heater_params: List[Any] = []
        mmi_params: List[Any] = []

        for layer in self.get_photonic_layers():
            lay_p: ClementsLayerParams = layer.get_params()
            blk_p: ClementsBlockParams = lay_p.clements_blk_params
            heater_params.append(blk_p.heater_params)
            mmi_params.append(blk_p.mmi_params)

        return ClementsMeshParams(
            n_inputs=n_inputs,
            n_layers=n_layers,
            is_matmul=self.get_is_matmul(),
            heater_params=heater_params,
            mmis_params=mmi_params,
        )

    # ------------------------------------------------------------------
    # Weight / phase snapshot helpers
    # ------------------------------------------------------------------

    def get_layer_phase_shifts(self) -> List[torch.Tensor]:
        """
        Return a list of phase tensors, one per layer.

        Each entry is the result of ClementsLayer.get_phase_shifts(), with
        shape [n_heaters_layer, n_heaters_per_block] as defined in the C++
        layer implementation.
        """
        phases_per_layer: List[torch.Tensor] = []

        for layer in self.get_photonic_layers():
            layer_phases: torch.Tensor = layer.get_phase_shifts()
            phases_per_layer.append(layer_phases)

        return phases_per_layer

    def get_weights_tensor(self) -> torch.Tensor:
        """
        Return a single 1D tensor with all heater phases in this mesh.

        Layers can have different [n_heaters_layer, n_heaters_per_block],
        so we flatten each layer and concatenate.
        """
        phases_per_layer = self.get_layer_phase_shifts()

        if not phases_per_layer:
            return torch.empty(0)

        flat_list: List[torch.Tensor] = []
        for ph in phases_per_layer:
            flat_list.append(ph.reshape(-1))

        return torch.cat(flat_list, dim=0)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_scalar_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.reshape(())
            return value.to(dtype=torch.float32).mean().reshape(())
        return lx_real(float(value))  # Dtype-aware scalar tensor

    def _get_layer_heater_metrics(self, layer):
        metrics = layer.get_heater_metrics()
        heater_list = metrics.get("heater_metrics", [])
        return heater_list, metrics

    def get_total_i_losses(self) -> torch.Tensor:
        P_in = self.get_input_optical_power()
        P_out = self.get_output_optical_power()
        self._total_i_losses = lx_real(10.0) * torch.log10(P_in / P_out)  # Dtype-aware 10.0
        return self._total_i_losses

    def get_heaters_quantization_error(self):
        mesh_errors = []

        for layer in self.get_photonic_layers():
            heater_list, _ = self._get_layer_heater_metrics(layer)
            heater_errors = [
                self._to_scalar_tensor(hm.get("quantization_error", 0.0))
                for hm in heater_list
            ]
            mesh_errors.append([heater_errors])

        return mesh_errors

    def get_heaters_snr(self) -> torch.Tensor:
        lst_snr = []

        for layer in self.get_photonic_layers():
            heater_list, _ = self._get_layer_heater_metrics(layer)
            snr_vals = [float(hm.get("snr_db", 0.0)) for hm in heater_list]
            if snr_vals:
                lst_snr.append(lx_real(snr_vals))  # Dtype-aware tensor from list

        if not lst_snr:
            return lx_real(0.0)  # Dtype-aware zero tensor

        return torch.stack(lst_snr, dim=0).mean(dim=0)

    def get_heaters_sqnr(self) -> torch.Tensor:
        lst_sqnr = []

        for layer in self.get_photonic_layers():
            heater_list, _ = self._get_layer_heater_metrics(layer)
            sqnr_vals = [float(hm.get("sqnr_db", 0.0)) for hm in heater_list]
            if sqnr_vals:
                lst_sqnr.append(lx_real(snr_vals))  # Dtype-aware tensor from list

        if not lst_sqnr:
            return lx_real(0.0)  # Dtype-aware zero tensor

        return torch.stack(lst_sqnr, dim=0).mean(dim=0)

    def get_power_consumption(self) -> torch.Tensor:
        total_power = lx_real(0.0)  # Dtype-aware accumulation

        for layer in self.get_photonic_layers():
            _, metrics = self._get_layer_heater_metrics(layer)
            total_power += lx_real(float(metrics.get("total_power_consumption", 0.0)))

        return total_power

    # ------------------------------------------------------------------
    # Metrics aggregation API
    # ------------------------------------------------------------------

    def compute_metrics(self) -> MeshMetrics:
        return MeshMetrics(
            total_i_losses=self.get_total_i_losses(),
            heaters_snr=self.get_heaters_snr(),
            heaters_sqnr=self.get_heaters_sqnr(),
            power_consumption=self.get_power_consumption(),
        )

    def get_metric(self, name: str) -> Any:
        metrics = self.compute_metrics()
        try:
            return getattr(metrics, name)
        except AttributeError as exc:
            available = list(MeshMetrics.__dataclass_fields__.keys())
            raise KeyError(
                f"Unknown metric '{name}'. Available metrics: {available}"
            ) from exc

    def compute_metrics_dict(self) -> Dict[str, Any]:
        m = self.compute_metrics()
        return {
            "total_i_losses": m.total_i_losses,
            "heaters_snr": m.heaters_snr,
            "heaters_sqnr": m.heaters_sqnr,
            "power_consumption": m.power_consumption,
        }
