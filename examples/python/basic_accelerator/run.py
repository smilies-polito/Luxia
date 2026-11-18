"""Minimal DAC-to-ADC Clements photonic pipeline with autograd."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from luxia.io_components import Adc, Dac
from luxia.params import (
    AdcParams,
    MmiParams,
    ClementsBlockParams,
    ClementsLayerParams,
    ClementsMeshParams,
    DacParams,
    HeaterParams,
    LaserModulatorParams,
)
from luxia.photonic_components import IdealPhotodetector, LaserModulator
from luxia.photonic_meshes.clements import ClementsMesh


def main() -> None:
    torch.manual_seed(7)
    n_inputs = 4
    x = torch.rand((1, n_inputs), requires_grad=True)

    dac = Dac(DacParams(number_of_bits=8))
    laser = LaserModulator(LaserModulatorParams(norm_input=False))
    mesh_params = ClementsMeshParams(
        number_of_inputs=n_inputs,
        number_of_layers=1,
        layer_params=[
            ClementsLayerParams(
                number_of_inputs=n_inputs,
                is_even_layer=True,
                clements_blk_params=ClementsBlockParams(
                    number_of_blocks=n_inputs // 2,
                    mmi_params=[
                        MmiParams(number_of_splitters=n_inputs // 2),
                        MmiParams(number_of_splitters=n_inputs // 2),
                    ],
                    heater_params=[
                        HeaterParams(number_of_heaters=n_inputs // 2),
                        HeaterParams(number_of_heaters=n_inputs // 2),
                    ],
                ),
            )
        ],
    )
    mesh = ClementsMesh(mesh_params)
    detector = IdealPhotodetector()
    adc = Adc(AdcParams(number_of_bits=8, max_analog_reference=float(n_inputs)))

    analog = dac(x)
    wave = laser(analog)
    mesh_output = mesh(wave)
    optical_power = detector(mesh_output)
    output = adc(optical_power)

    print(f"Input:                 {x.detach()}")
    print(f"After DAC:             {analog.detach()}")
    print(f"After laser modulator: {wave.detach()}")
    print(f"After Clements mesh:   {mesh_output.detach()}")
    print(f"After photodetector:   {optical_power.detach()}")
    print(f"After ADC:             {output.detach()}")

    output.sum().backward()
    print(f"Input gradient:        {x.grad}")


if __name__ == "__main__":
    main()
