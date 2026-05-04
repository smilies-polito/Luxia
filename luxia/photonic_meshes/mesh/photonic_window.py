import torch
from torch import nn

from luxia import LxModule
from luxia.photonic_meshes.mesh import PhotonicLayer


class PhotonicWindow(LxModule):
    """
    A computational window over a contiguous group of PhotonicLayer objects.

    Semantics:
    - A window contains one or more consecutive layers.
    - Single layer: dispatch directly to that layer.
    - Multiple layers: compose the window transfer matrix and apply it once.

    Note:
    - `window_size` belongs to PhotonicMesh and means "number of layers per window".
    - Each window computes its own local TF matrix (when multi-layer).
    - No sequential execution inside windows.
    """

    def __init__(
        self,
        layer_list: list[PhotonicLayer],
        debug: bool = False,
    ):
        super().__init__(debug=debug)

        if layer_list is None or len(layer_list) == 0:
            raise ValueError("PhotonicWindow requires at least one PhotonicLayer")

        for i, layer in enumerate(layer_list):
            if not isinstance(layer, PhotonicLayer):
                raise TypeError(
                    f"layer_list[{i}] must be a PhotonicLayer, got {type(layer)}"
                )

        self._layers = nn.ModuleList(layer_list)
        self._n_layers = len(layer_list)
        self._n_inputs = layer_list[0].get_n_inputs()

        for i, layer in enumerate(layer_list):
            if layer.get_n_inputs() != self._n_inputs:
                raise ValueError(
                    "All layers in a PhotonicWindow must have the same n_inputs. "
                    f"Expected {self._n_inputs}, got {layer.get_n_inputs()} at index {i}"
                )

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        self._check_input(x_complex)

        # Single layer: fast path
        if self._n_layers == 1:
            return self._layers[0](x_complex)

        # Multi-layer: compose TF matrix and apply
        return self.matmul_forward(x_complex)

    def matmul_forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        tf_matrix = self.get_tf_matrix()
        return torch.matmul(x_complex, tf_matrix)

    def get_tf_matrix(self) -> torch.Tensor:
        return torch.linalg.multi_dot(
            tuple(layer.get_tf_matrix() for layer in self._layers)
        )

    def _check_input(self, x_complex: torch.Tensor):
        if x_complex.dim() != 2:
            raise ValueError(
                "PhotonicWindow expects a tensor of shape [batch, n_inputs], "
                f"got {tuple(x_complex.shape)}"
            )

        if x_complex.shape[1] != self._n_inputs:
            raise ValueError(
                f"PhotonicWindow expected {self._n_inputs} inputs, "
                f"got {x_complex.shape[1]}"
            )

    def get_layers(self):
        return self._layers

    def get_n_layers(self):
        return self._n_layers

    def get_n_inputs(self):
        return self._n_inputs
