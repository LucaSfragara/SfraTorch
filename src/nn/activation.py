import numpy as np
import scipy
import scipy.special


class Identity:
    """
    Identity activation function.
    """

    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ

class Sigmoid:
    """
    Sigmoid activation function
    """
    def forward(self, Z):

        self.A = Z
        self.npVal = np.exp(-self.A)
        return 1 / (1 + self.npVal)

    def backward(self, dLdA):

        dAdZ = self.npVal / (1 + self.npVal) ** 2
        return dAdZ*dLdA

class Tanh:
   
    def forward(self, Z):

        self.A = Z
        self.tanhVal =  np.tanh(self.A)
        return self.tanhVal

    def backward(self, dLdA, state=None):
        if state is not None:
            dAdZ = 1 - state*state
            return dAdZ * dLdA
        else:
            dAdZ = 1 - self.tanhVal * self.tanhVal
            return dAdZ * dLdA



class ReLU:
    """
    ReLU (Rectified Linear Unit) activation function.
    """
    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = np.where(self.A>0, 1, 0)
        dLdZ = dLdA * dAdZ
        return dLdZ
    

class GELU:
    """
    GELU (Gaussian Error Linear Unit) activation function.
    """
    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.Z = Z
        self.A = 0.5*Z*(1+scipy.special.erf(Z/np.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = 0.5 *(1+scipy.special.erf(self.Z/np.sqrt(2))) + (self.Z/np.sqrt(2*np.pi)*np.exp(-self.Z**2/2))
        dLdZ = dLdA * dAdZ
        return dLdZ


class Softmax:
    """
    Softmax activation function.
    """

    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.Z = Z
        Z_stable = Z- np.max(Z, axis = 1, keepdims=True)
        self.A = np.exp(Z_stable)/ np.sum(np.exp(Z_stable), axis=1, keepdims=True)
        return self.A  

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        N = self.Z.shape[0]
        C = self.Z.shape[1]

        dLdZ = np.zeros((N, C))

        for i in range(N):
            J = np.zeros((C, C))

            for m in range(C):
                for n in range(C):
                    a_m = self.A[i, m]
                    a_n = self.A[i, n]
                    if m == n:
                        J[m, n] = a_m*(1-a_m)
                    else:
                        J[m, n] = -a_m*a_n

            dLdZ[i, :] = np.dot(J, dLdA[i, :])

        return dLdZ
