import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """
    def __init__(self):
        """
        Initialize the ScaledDotProductAttention class.
        """
        self.eps = 1e10
        self.softmax = Softmax(dim=-1)


    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        self.Q = Q
        self.K = K
        scaled_dot_product = (Q @ np.swapaxes(K, -1, -2))/np.sqrt(Q.shape[-1])

        if mask is not None:
            scaled_dot_product[mask] = -self.eps

        self.attention_scores = self.softmax.forward(scaled_dot_product)

        self.V = V
        output = self.attention_scores @ V

        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        d_V = np.swapaxes(self.attention_scores, -1, -2) @ d_output

        d_attention_scores = d_output @ np.swapaxes(self.V, -1, -2)
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)

        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(self.Q.shape[-1])

        d_Q = d_scaled_dot_product @ self.K
        d_K = np.swapaxes(d_scaled_dot_product, -1, -2) @ self.Q

        return d_Q, d_K, d_V
