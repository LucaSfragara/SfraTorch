import numpy as np


class BatchNorm1d:
    """
    Batch Normalization layer for 1D inputs.
    """

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))
        self.C = num_features
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        Forward pass for batch normalization.
        :param Z: batch of input data Z (N, num_features).
        :param eval: flag to indicate training or inference mode.
        :return: batch normalized data.
        """
        self.Z = Z
        self.N = self.Z.shape[0]
        self.M = 1/self.N * np.sum(self.Z, axis = 0, keepdims= True)
        self.V = 1/self.N * np.sum((self.Z-self.M)**2, axis = 0, keepdims= True)

        if eval == False:
            self.NZ = (self.Z  - self.M)/np.sqrt(self.V + self.eps)
            self.BZ = (self.BW + np.zeros((self.N, self.C))) * self.NZ + (self.Bb + np.zeros((self.N, self.C)))

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else:
            self.NZ = (self.Z  - self.running_M)/np.sqrt(self.running_V + self.eps)
            self.BZ = (self.BW + np.zeros((self.N, self.C))) * self.NZ + (self.Bb + np.zeros((self.N, self.C)))

        return self.BZ

    def backward(self, dLdBZ):
        """
        Backward pass for batch normalization.
        :param dLdBZ: Gradient loss wrt the output of BatchNorm transformation for Z (N, num_features).
        :return: Gradient of loss (L) wrt batch of input batch data Z (N, num_features).
        """
        N, C = dLdBZ.shape

        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)

        dLdNZ = dLdBZ * self.BW
        dLdV = np.sum(dLdNZ * (self.Z - self.M) * (-0.5) * (self.V + self.eps) ** (-1.5), axis=0, keepdims=True)
        dNZdM = -1 / np.sqrt(self.V + self.eps) - (2 / N) * np.sum(self.Z - self.M, axis=0, keepdims=True)
        dLdM = np.sum(dLdNZ * dNZdM, axis=0, keepdims=True)
        dLdZ = dLdNZ * (1 / np.sqrt(self.V + self.eps)) + (dLdV * 2 * (self.Z - self.M) / N) + (dLdM / N)

        return dLdZ 