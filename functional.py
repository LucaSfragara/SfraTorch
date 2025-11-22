
import numpy as np
from .autograd_engine import Autograd


def identity_backward(grad_output, a):
    """Backward for identity."""
    return grad_output


def add_backward(grad_output, a, b):
    """Backward for addition."""
    a_grad = grad_output * np.ones(a.shape)
    b_grad = grad_output * np.ones(b.shape)

    return a_grad, b_grad


def sub_backward(grad_output, a, b):
    """Backward for subtraction"""

    a_grad = grad_output * np.ones(a.shape)
    b_grad = -grad_output * np.ones(b.shape)

    return a_grad, b_grad


def matmul_backward(grad_output, a, b):
    """Backward for matrix multiplication"""
    
    a_grad = grad_output @ b.T
    b_grad = a.T @ grad_output

    return a_grad, b_grad


def mul_backward(grad_output, a, b):
    """Backward for multiplication"""

    a_grad = grad_output * b
    b_grad = grad_output * a

    return a_grad, b_grad


def div_backward(grad_output, a, b):
    """Backward for division"""

    a_grad = grad_output / b
    b_grad = -grad_output * a / (b ** 2)

    return a_grad, b_grad



def log_backward(grad_output, a):
    """Backward for log"""

    a_grad = grad_output / a

    return a_grad


def exp_backward(grad_output, a):
    """Backward of exponential"""

    a_grad = grad_output * np.exp(a)

    return a_grad

def exp_backward_neg(grad_output, a):
    """Backward of exponential"""

    a_grad = -grad_output * np.exp(-a)

    return a_grad

def max_backward(grad_output, a):
    """Backward of max"""
    max_val = np.max(a)

    a_grad = grad_output * (a>0)
    return a_grad


def sum_backward(grad_output, a):
    """Backward of sum"""
    a_grad = grad_output * np.ones(a.shape)
    return a_grad
  
def tanh_backward(grad_output, a):
    """Backward for tanh."""
    t = np.tanh(a)
    a_grad = grad_output * (1 - t * t)
    return a_grad


def SoftmaxCrossEntropy_backward(grad_output, pred, ground_truth):
    """Backward for Softmax CrossEntropy Loss."""
    exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
    softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

    batch_size = pred.shape[0]

    dpred = (softmax_pred - ground_truth) / batch_size

    dpred = dpred
    return dpred, None
