import numpy as np
import sys

from src.rnn_cell import *
from src.nn.linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = [
             RNNCell(input_size, hidden_size) if i == 0
                 else RNNCell(hidden_size, hidden_size)
                     for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.

        Parameters
        ----------
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits: (batch_size, output_size) 

        Output (y): logits

        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        self.x = x
        self.hiddens.append(hidden.copy())
        logits = None

        for t in range(seq_len):
            h_t = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype = float)
            for layer in range(self.num_layers):
                rnn_cell = self.rnn[layer]
                if layer == 0:
                    input_t = x[:, t]
                else:
                    input_t = h_t[layer-1]
                h_t_l = rnn_cell.forward(input_t, h_prev_t= self.hiddens[t][layer])
                h_t[layer, :, :] = h_t_l

            self.hiddens.append(h_t)

        A = self.hiddens[-1][-1]
        logits = self.output_layer.forward(A)

        return logits

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Parameters
        ----------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                gradient w.r.t. the last time step output.

        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        new_dh = np.zeros_like(dh)
        for t in reversed(range(seq_len)):
            for layer in reversed(range(self.num_layers)):
                h_prev_l = self.hiddens[t+1][layer-1] if layer != 0 else self.x[:, t]
                h_prev_t = self.hiddens[t][layer]
                h_t = self.hiddens[t+1][layer]
                rnn_cell = self.rnn[layer]

                dx, dh_prev_t =  rnn_cell.backward(dh[layer], h_t, h_prev_l, h_prev_t)
                new_dh[layer] = dh_prev_t

                if layer != 0:
                     dh[layer - 1] += dx

            dh = new_dh

        return dh / batch_size
