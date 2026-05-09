from neuroptica.components import MZI, PhaseShifter
import neuroptica as neu
from typing import Type
import numpy as np
from neuroptica.layers import OpticalMeshNetworkLayer
from neuroptica.losses import Loss
from neuroptica.models import Sequential
import wandb

import time
from util.mem_usage import get_cpu_mem_usage


class ForkedOptimizerGD(neu.InSituGradientDescent):
    '''
    ForkedOptimizerGD is a gradient descent optimizer that forked from
    the original Gradient Descent optimizer in the Neuroptica library.
    '''

    def __init__(self, model: Sequential,
                 loss: Type[Loss], learning_rate=0.01):

        super().__init__(model, loss, learning_rate)

    def fit(self, data: np.ndarray, labels: np.ndarray,
            data_val: np.ndarray, labels_val: np.ndarray,
            epochs=1000, batch_size=32, verbose=True):
        avg_loss = []
        avg_vacc = []
        times = []
        mem_cpu = []

        n_features, n_samples = data.shape
        for epoch in range(epochs):
            start = time.time()  # start counting time
            avg_epoch_loss = self.fit_epoch(data, labels, batch_size)
            epoch_val_acc = self.fit_val_epoch(
                data_val, labels_val, batch_size)
            deltat = time.time()-start  # end counting time
            mem = get_cpu_mem_usage(verbose=verbose)

            avg_loss.append(avg_epoch_loss)
            avg_vacc.append(epoch_val_acc)
            times.append(deltat)
            mem_cpu.append(mem)

            if verbose:
                print(f'LOSS train {avg_epoch_loss}')
                print(f'ACC train valid {epoch_val_acc}')
                print(f'ELAPSED TIME {deltat} seconds')
                print(flush=True)
                print()

        return avg_loss, avg_vacc, times, mem_cpu

    def fit_epoch(self, data: np.ndarray, labels: np.ndarray, batch_size=32):

        epoch_loss = 0.0
        for X, Y in self.make_batches(data, labels, batch_size):
            # Propagate the data forward
            Y_hat = self.model.forward_pass(data)
            d_loss = self.loss.dL(Y_hat, labels)
            epoch_loss += self.loss.L(Y_hat, labels)

            # Compute the backpropagated signals for the model
            gradients = self.model.backward_pass(d_loss)
            delta_prev = d_loss  # backprop signal to send in the final layer

            # Compute the foward and adjoint fields at each
            # phase shifter in all tunable layers
            for layer in reversed(self.model.layers):
                if isinstance(layer, OpticalMeshNetworkLayer):
                    layer.mesh.adjoint_optimize(
                        layer.input_prev, delta_prev,
                        lambda dx: -1 * self.learning_rate * dx)

                # Set the backprop signal for the subsequent
                # (spatially previous) layer
                delta_prev = gradients[layer.__name__]

            avg_loss_epoch = epoch_loss / labels.shape[1]
        return avg_loss_epoch

    def fit_val_epoch(self, data_val: np.ndarray,
                      labels_val: np.ndarray, batch_size=32):
        acc = []
        for X_val, Y_val in self.make_batches(data_val, labels_val, batch_size):
            Y_pred = self.model.forward_pass(X_val)
            b_acc = np.mean(np.argmax(Y_pred, axis=0)
                            == np.argmax(Y_val, axis=0))
            acc.append(b_acc)
        return np.mean(acc)


class ForkedOptimizerAdam(neu.InSituAdam):
    def __init__(self, model: Sequential, loss: Type[Loss],
                 learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8):
        super().__init__(model, loss, learning_rate, beta1, beta2, epsilon)

    def fit(self, data: np.ndarray, labels: np.ndarray,
            data_val: np.ndarray, labels_val: np.ndarray,
            epochs=1000, batch_size=32, verbose=True,
            cache_fields=False, use_partial_vectors=False):
        '''
        ForkedOptimizerAdam is a forked version of the original InSituAdam
        optimizer in the Neuroptica library. This optimizer is used to train
        the model with the Adam optimizer.
        '''
        avg_loss = []
        avg_vacc = []
        times = []
        mem_cpu = []

        n_features, n_samples = data.shape
        for epoch in range(epochs):
            if verbose:
                print(f'EPOCH {epoch+1}:')

            start = time.time()  # start counting time
            avg_epoch_loss = self.fit_epoch(
                data, labels, batch_size,
                cache_fields=cache_fields,
                use_partial_vectors=use_partial_vectors)

            epoch_val_acc = self.fit_val_epoch(
                data_val, labels_val, batch_size)

            deltat = time.time()-start  # end counting time
            mem = get_cpu_mem_usage(verbose=verbose)

            avg_loss.append(avg_epoch_loss)
            avg_vacc.append(epoch_val_acc)
            times.append(deltat)
            mem_cpu.append(mem)

            if verbose:
                print(f'LOSS train {avg_epoch_loss}')
                print(f'ACC train valid {epoch_val_acc}')
                print(f'ELAPSED TIME {deltat} seconds')
                print(flush=True)
                print()

            wandb.log({
                'train_loss': avg_epoch_loss,
                'val_accuracy': epoch_val_acc,
                'time': deltat,
            })

        return avg_loss, avg_vacc, times, mem_cpu

    def fit_epoch(self, data: np.ndarray, labels: np.ndarray, batch_size=32,
                  cache_fields=False, use_partial_vectors=False):
        epoch_loss = 0.0
        batch = 0
        for X, Y in self.make_batches(data, labels, batch_size):
            batch += 1
            self.t += 1

            # Propagate the data forward
            Y_hat = self.model.forward_pass(
                X, cache_fields=cache_fields,
                use_partial_vectors=use_partial_vectors)

            d_loss = self.loss.dL(Y_hat, Y)
            epoch_loss += np.sum(self.loss.L(Y_hat, Y))

            # Compute the backpropagated signals for the model
            deltas = self.model.backward_pass(
                d_loss,
                cache_fields=cache_fields,
                use_partial_vectors=use_partial_vectors)

            delta_prev = d_loss  # backprop signal to send in the final layer

            # Compute the foward and adjoint fields at each phase shifter 
            # in all tunable layers
            for layer in reversed(self.model.layers):
                if isinstance(layer, OpticalMeshNetworkLayer):
                    gradients = layer.mesh.compute_gradients(
                        layer.input_prev, delta_prev,
                        cache_fields=cache_fields,
                        use_partial_vectors=use_partial_vectors)

                    for cmpt in gradients:
                        self.g[cmpt] = np.mean(gradients[cmpt], axis=-1)
                        self.m[cmpt] = self.beta1 * self.m[cmpt] + \
                            (1 - self.beta1) * self.g[cmpt]
                        self.v[cmpt] = self.beta2 * self.v[cmpt] + \
                            (1 - self.beta2) * self.g[cmpt] ** 2
                        mhat = self.m[cmpt] / (1 - self.beta1 ** self.t)
                        vhat = self.v[cmpt] / (1 - self.beta2 ** self.t)

                        grad = -1 * self.step_size * mhat / \
                            (np.sqrt(vhat) + self.epsilon)

                        # Adjust settings by gradient amount
                        if isinstance(cmpt, PhaseShifter):
                            cmpt.phi += grad[0]

                        elif isinstance(cmpt, MZI):
                            dtheta, dphi = grad
                            cmpt.theta += dtheta
                            cmpt.phi += dphi

                # Set the backprop signal for the subsequent
                # (spatially previous) layer
                delta_prev = deltas[layer.__name__]

        avg_epoch_loss = epoch_loss / labels.shape[1]
        return avg_epoch_loss

    def fit_val_epoch(self, data_val: np.ndarray,
                      labels_val: np.ndarray, batch_size=32):
        acc = []
        for X_val, Y_val in self.make_batches(data_val, labels_val, batch_size):
            Y_pred = self.model.forward_pass(X_val)
            b_acc = np.mean(np.argmax(Y_pred, axis=0)
                            == np.argmax(Y_val, axis=0))
            acc.append(b_acc)
        return np.mean(acc)
