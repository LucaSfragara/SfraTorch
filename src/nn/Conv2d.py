import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        
        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))
        
        for i in range(output_height):
            for j in range(output_width):
                window = A[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                Z[:, :, i, j] = np.tensordot(window, self.W, axes=([1, 2, 3], [1, 2, 3]))

        Z = Z + self.b.reshape(1, -1, 1, 1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        _, _, output_height, output_width = dLdZ.shape
        _, _,  input_height, input_width = self.A.shape
        
        self.dLdW = np.zeros_like(self.W)
        for i in range(output_height):
            for j in range(output_width):
                window = self.A[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                self.dLdW += np.tensordot(dLdZ[:, :, i, j], window, axes=([0], [0]))
        
        
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))
        
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)))
        W_flipped = np.flip(np.flip(self.W, axis=2), axis=3)        
        
        dLdA = np.zeros_like(self.A)
        
        for i in range(input_height):
            for j in range(input_width):
                window = dLdZ_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                dLdA[:, :, i, j] = np.tensordot(window, W_flipped, axes=([1,2,3], [0,2,3]))
        
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding

        self.conv2d_stride1 = Conv2d_stride1(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             weight_init_fn=weight_init_fn,
                                             bias_init_fn=bias_init_fn)
        self.downsample2d = Downsample2d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)))
        Z = self.conv2d_stride1.forward(A_padded)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA =  self.conv2d_stride1.backward(dLdZ)
        dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dLdA
