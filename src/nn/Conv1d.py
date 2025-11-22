import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1
        
        Z = np.zeros((batch_size, self.out_channels, output_size))

        for i in range(output_size):
            window = A[:, :, i:i+self.kernel_size]
            Z[:, :, i] = np.tensordot(window, self.W, axes=([1, 2], [1, 2]))

        Z = Z + self.b.reshape(1, -1, 1)
    
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, _, output_size = dLdZ.shape

        dLdW = np.zeros_like(self.W)
        for i in range(output_size):
            window = self.A[:, :, i:i+self.kernel_size]
            dLdW += np.tensordot(dLdZ[:, :, i], window, axes=([0], [0]))
        self.dLdW = dLdW

        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        input_size = self.A.shape[2]
        dLdA = np.zeros_like(self.A)
        pad_width = self.kernel_size - 1
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (pad_width, pad_width)))
        flipped_w = np.flip(self.W, axis=2)

        for i in range(input_size):
            window = dLdZ_padded[:, :, i:i + self.kernel_size]
            dLdA[:, :, i] = np.tensordot(window, flipped_w, axes=([1, 2], [0, 2]))

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding

        self.conv1d_stride1 = Conv1d_stride1(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                            weight_init_fn=weight_init_fn,
                                            bias_init_fn=bias_init_fn)

        self.downsample1d = Downsample1d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)))
        Z_conv = self.conv1d_stride1.forward(A_padded)
        Z = self.downsample1d.forward(Z_conv)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        dLdZ_conv = self.downsample1d.backward(dLdZ)
        dLdA = self.conv1d_stride1.backward(dLdZ_conv)

        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:-self.pad]

        return dLdA
