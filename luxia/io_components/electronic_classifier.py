import torch.nn as nn
from luxia import LxModule
import luxia.util.log_config as lx_log

class ElectronicClassifier(LxModule):
    def __init__(self, n_inputs: int, n_classes: int, debug=False):
        super().__init__(debug=debug)
        self.classifier = nn.Linear(n_inputs, n_classes)

    def forward(self, x_reals):
        y = self.classifier(x_reals)
        if self._debug:
            lx_log.setLevel("DEBUG")
            lx_log.debug(f"Input to classifier: \n{x_reals}")
            lx_log.debug(f"Output from classifier: \n{y}")
        return y

