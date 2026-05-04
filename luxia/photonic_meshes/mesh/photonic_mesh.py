from .photonic_layer import PhotonicLayer
from .photonic_window import PhotonicWindow
from luxia import LxModule
from luxia.util.log_config import lx_log
from torch import nn
import torch


def _heatmap_mesh_usage(values: torch.Tensor, max_cols: int = 80) -> str:
    """
    Render a 2D tensor as ASCII heatmap.
    Rows = samples, cols = channels (optionally downsampled horizontally).
    Expects values already in [0, 1].
    """
    if values.dim() == 1:
        v = values.unsqueeze(0)
    elif values.dim() > 2:
        v = values.flatten(0, -2)
    else:
        v = values

    v = v.detach().float()
    rows, cols = v.shape

    if cols > max_cols:
        step = max(cols // max_cols, 1)
        v = v[:, ::step]

    v = torch.clamp(v, 0.0, 1.0)

    chars = " ▁▂▃▄▅▆▇█"
    lines = []

    for row in v:
        line = ""
        for x in row:
            x = float(x)
            lvl = int(x * (len(chars) - 1))
            c = chars[lvl]

            if x < 0.1:
                color = 34
            elif x < 0.4:
                color = 32
            elif x < 0.7:
                color = 31
            else:
                color = 37

            line += f"\\x1b[1;{color}m{c}\\x1b[0m"
        lines.append(line)

    return "\n".join(lines)


class PhotonicMesh(LxModule):
    def __init__(
        self,
        n_inputs: int,
        n_layers: int,
        is_matmul: bool = False,
        layer_list: list[PhotonicLayer] | None = None,
        window_size: int = 1,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        
        if layer_list is None:
            raise ValueError("You must provide a list of PhotonicLayer objects")

        if len(layer_list) != n_layers:
            raise ValueError(
                "layer_list must have the same number of layers as n_layers"
            )

        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self._is_matmul = is_matmul
        self._n_inputs = n_inputs
        self._n_layers = n_layers
        self._window_size = window_size
        self._input_op_power = 0.0
        self._output_op_power = 0.0

        for i, layer in enumerate(layer_list):
            if not isinstance(layer, PhotonicLayer):
                raise TypeError(
                    f"layer_list[{i}] must be a PhotonicLayer, got {type(layer)}"
                )
            if layer.get_n_inputs() != n_inputs:
                raise ValueError(
                    f"layer_list[{i}] has n_inputs={layer.get_n_inputs()}, "
                    f"expected {n_inputs}"
                )

        self._ph_layers = nn.ModuleList(layer_list)
        self._ph_windows = self._build_windows(layer_list, window_size)

    def _build_windows(
        self,
        layer_list: list[PhotonicLayer],
        window_size: int,
    ) -> nn.ModuleList:
        windows = []

        for i in range(0, len(layer_list), window_size):
            chunk = layer_list[i:i + window_size]

            windows.append(
                PhotonicWindow(
                    layer_list=chunk,
                    debug=self._debug,
                )
            )

        return nn.ModuleList(windows)

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        self._input_op_power = torch.sum(torch.abs(x_complex) ** 2)

        if self._is_matmul:
            y_complex = self.matmul_forward(x_complex)
        else:
            y_complex = self.window_forward(x_complex)

        self._output_op_power = torch.sum(torch.abs(y_complex) ** 2)

        if self._debug:
            self._debug_forward(x_complex, y_complex)

        return y_complex

    def window_forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        y_complex = x_complex

        for window in self._ph_windows:
            y_complex = window(y_complex)

        return y_complex

    def matmul_forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        tf_matrix = self.get_tf_matrix()
        return torch.matmul(x_complex, tf_matrix)

    def get_tf_matrix(self):
        if len(self._ph_layers) == 1:
            return self._ph_layers[0].get_tf_matrix()

        return torch.linalg.multi_dot(
            tuple(layer.get_tf_matrix() for layer in self._ph_layers)
        )

    def _debug_forward(self, x_complex: torch.Tensor, y_complex: torch.Tensor):
        torch.set_printoptions(profile="full")
        lx_log.setLevel("DEBUG")
        n_show = 10

        in_power = torch.abs(x_complex) ** 2
        out_power = torch.abs(y_complex) ** 2

        B, N_out = out_power.shape
        n_show = min(n_show, B)

        in_power = in_power[:n_show]
        out_power = out_power[:n_show]

        x_first = x_complex[0]
        y_first = y_complex[0]

        lx_log.debug("=== PhotonicMesh debug ===")
        lx_log.debug(f"x (first vec):\n{x_first}")
        lx_log.debug(f"|x|^2 (first vec):\n{torch.abs(x_first) ** 2}")
        lx_log.debug(f"sum(|x|^2): {torch.sum(torch.abs(x_first) ** 2)}")

        lx_log.debug(f"y (first vec):\n{y_first}")
        lx_log.debug(f"|y|^2 (first vec):\n{torch.abs(y_first) ** 2}")
        lx_log.debug(f"sum(|y|^2): {torch.sum(torch.abs(y_first) ** 2)}")

        lx_log.debug(f"Total input power (batch): {self._input_op_power}")
        lx_log.debug(f"Total output power (batch): {self._output_op_power}")
        lx_log.debug(f"Window size: {self._window_size}")
        lx_log.debug(f"Number of windows: {len(self._ph_windows)}")
        lx_log.debug(f"Global is_matmul: {self._is_matmul}")

        total_in_per_sample = in_power.sum(dim=1, keepdim=True)
        eps = 1e-20

        frac = out_power / (total_in_per_sample + eps)

        log_frac = torch.log10(frac + eps)
        log_min, log_max = -6.0, 0.0
        log_frac = torch.clamp(log_frac, log_min, log_max)

        frac_norm = (log_frac - log_min) / (log_max - log_min)

        ref_col = torch.ones(
            (n_show, 1),
            device=frac_norm.device,
            dtype=frac_norm.dtype,
        )

        heatmap_mat = torch.cat([ref_col, frac_norm], dim=1)

        lx_log.debug(
            f"Heatmap rows = {n_show}; "
            "col 0 = input ref; cols 1.. = |y|^2_out / total_input_power (log scale)"
        )
        lx_log.debug("\n" + _heatmap_mesh_usage(heatmap_mat, max_cols=80))

        torch.set_printoptions(profile="default")

    def get_photonic_layers(self):
        return self._ph_layers

    def get_photonic_windows(self):
        return self._ph_windows

    def get_input_optical_power(self):
        return self._input_op_power

    def get_output_optical_power(self):
        return self._output_op_power

    def get_n_inputs(self):
        return self._n_inputs

    def get_n_layers(self):
        return self._n_layers

    def get_window_size(self):
        return self._window_size

    def get_n_windows(self):
        return len(self._ph_windows)

    def get_is_matmul(self):
        return self._is_matmul
