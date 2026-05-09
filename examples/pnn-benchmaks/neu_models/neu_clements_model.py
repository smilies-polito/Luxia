import neuroptica as neu


class NeuClementsModel(neu.Sequential):
    def __init__(self,
                 n_inputs: int,
                 n_layers: int,
                 n_classes: int,
                 n_architectures=2):

        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_arch = n_architectures

        list_seq = []

        for i in range(n_architectures):
            list_seq.append(neu.ClementsLayer(
                N=self.n_inputs, M=self.n_layers)
            )

            list_seq.append(neu.Activation(
                neu.AbsSquared(N=self.n_inputs)
            )
            )
        lower_limit, upper_limit = self._getLimits(
            self.n_inputs, self.n_classes)

        list_seq.append(neu.DropMask(
            self.n_inputs, keep_ports=range(lower_limit, upper_limit)))
        super(NeuClementsModel, self).__init__(list_seq)

    def _getLimits(self, n_inputs, n_classes):
        """
        given the number of inputs and the number of classes it returns the
        lower and upper limits, which are used to slice the output of the model
        this limits are used to get the middle part of the output of the model

        Args:
            n_inputs (int): number of inputs
            n_classes (int): number of classes

        Returns: lower_limit, upper_limit
        """
        lower_limit = n_inputs//2 - (n_classes//2)
        upper_limit = n_inputs//2 + n_classes//2

        if n_classes % 2 != 0:
            upper_limit = upper_limit + 1

        return lower_limit, upper_limit
