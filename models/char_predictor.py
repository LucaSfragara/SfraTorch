import numpy as np
import sys

from src.gru_cell import *
from src.nn.linear import *


class CharacterPredictor(object):
    """CharacterPredictor class."""

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        self.gru = GRUCell(input_dim, hidden_dim)
        self.projection = Linear(hidden_dim,num_classes)
        self.num_classes =  num_classes
        self.hidden_dim = hidden_dim
        self.projection.W = np.random.rand(num_classes, hidden_dim) #type: ignore

    def init_rnn_weights(
        self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
    ):
        self.gru.init_weights(
            Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
        )

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """CharacterPredictor forward.

        A pass through one time step of the input

        Input
        -----
        x: (feature_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.
        
        Returns
        -------
        logits: (num_classes)
            hidden state at current time-step.

        hnext: (hidden_dim)
            hidden state at current time-step.

        """
        hnext = self.gru.forward(x, h)
        hnext_proj = hnext.reshape(1, -1)
        logits = self.projection.forward(hnext_proj)
        logits = logits.reshape(-1,)

        return logits, hnext


def inference(net:CharacterPredictor, inputs:np.ndarray):
    """CharacterPredictor inference.

    An instance of the class defined above runs through a sequence of inputs to
    generate the logits for all the timesteps.

    Input
    -----
    net:
        An instance of CharacterPredictor.

    inputs: (seq_len, feature_dim)
            a sequence of inputs of dimensions.

    Returns
    -------
    logits: (seq_len, num_classes)
            one per time step of input..

    """
    h = np.zeros((net.hidden_dim,))
    output = np.empty((len(inputs), net.num_classes))

    for t in range(len(inputs)):
        logits, h = net.forward(x = inputs[t], h  = h)
        output[t] = logits

    return output
