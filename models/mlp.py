import numpy as np

from src.nn.linear import Linear
from src.nn.activation import ReLU


class MLP0:
    def __init__(self, debug=False):
        """
        Initialize a single linear layer of shape (2,3).
        Use ReLU activations for the layer.
        """
        self.debug = debug
        self.layers = [Linear(2, 3), ReLU()]

    def forward(self, A0):
        """
        Forward pass through the linear layer followed by the activation layer.
        """
        l = len(self.layers)
        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)
        
        if self.debug:
            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):
        """
        Backward pass through the model.
        """
        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP1:
    def __init__(self, debug=False):
        """
        Initialize 2 linear layers.
        Layer 1 of shape (2,3).
        Layer 2 of shape (3, 2).
        Use ReLU activations for both the layers.
        Implement it on the same lines (in a list) as in the MLP0 class.
        """
        self.debug = debug
        self.layers = [Linear(2, 3), ReLU(), Linear(3, 2), ReLU()]

    def forward(self, A0):
        """
        Forward pass through the linear layers and corresponding activation layers.
        """
        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)

        Z1 = self.layers[2].forward(A1)
        A2 = self.layers[3].forward(Z1)

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):
        """
        Backward pass through the model.
        """
        dLdZ1 = self.layers[3].backward(dLdA2)  
        dLdA1 = self.layers[2].backward(dLdZ1)

        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:
            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0

class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2).
        """
        self.debug = debug
        self.layers = [Linear(2, 4), ReLU(), Linear(4, 8), ReLU(), Linear(8, 8), ReLU(), Linear(8, 4), ReLU(), Linear(4, 2), ReLU()]

    def forward(self, A):
        """
        Forward pass through the linear layers and corresponding activation layers.
        """
        if self.debug:
            self.A = [A]

        L = len(self.layers)

        for i in range(L):
            A = self.layers[i].forward(A)
            
            if self.debug:
                self.A.append(A)

        return A

    def backward(self, dLdA):
        """
        Backward pass through the model.
        """
        if self.debug:
            self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):
            dLdA = self.layers[i].backward(dLdA)
            if self.debug:
                self.dLdA = [dLdA] + self.dLdA

        return  dLdA
