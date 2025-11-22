import numpy as np
import os
from .activation import Softmax

class Criterion(object):
    """
    Interface for loss functions.
    """
    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """

        self.logits = x
        self.labels = y
        self.batch_size = self.labels.shape[0]
        exps = np.exp(self.logits)
        self.softmax = exps / exps.sum(axis=1, keepdims=True)
        self.loss = np.sum(np.multiply(self.labels, -np.log(self.softmax)), axis=1)

        return self.loss

    def backward(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        self.gradient = self.softmax - self.labels

        return self.gradient

class MSELoss:
    def forward(self, A, Y):
        """
        Calculate the Mean Squared error (MSE)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss (scalar)
        """
        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]
        self.C = self.A.shape[1]
        se = (A-Y) * (A-Y) # (N, C)
        sse = np.ones((self.N, 1)).T @ se @ np.ones((self.C, 1))
        mse = sse/(self.N*self.C)
        return mse  

    def backward(self):
        """
        Calculate the gradient of MSE Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.
        """
        dLdA = 2* (self.A-self.Y)/(self.N*self.C)
        return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss (XENT)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss (scalar)
        """
        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]
        self.C = self.Y.shape[1]

        Ones_C = np.ones((self.C, 1))
        Ones_N = np.ones((self.N, 1))

        self.softmax = Softmax().forward

        crossentropy = -Y * np.log(self.softmax(self.A)) @ Ones_C #
        sum_crossentropy_loss = np.sum(crossentropy)
        mean_crossentropy_loss = sum_crossentropy_loss / self.N

        return mean_crossentropy_loss
    
    def backward(self):
        """
        Calculate the gradient of Cross-Entropy Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.
        """
        dLdA = (self.softmax(self.A)-self.Y)/self.N
        return dLdA