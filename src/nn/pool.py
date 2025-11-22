
import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        out_h = A.shape[2] - self.kernel + 1
        out_w = A.shape[3] - self.kernel + 1
        Z = np.zeros((A.shape[0], A.shape[1], out_h, out_w))
        self.A = A
        for i in range(out_h):
            for j in range(out_w):
                window = A[:, :, i:i+self.kernel, j:j+self.kernel]
                Z[:, :, i, j] = np.max(window, axis=(2, 3))
            
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, num_channels, out_height, out_width = dLdZ.shape
        input_height = out_height + self.kernel - 1
        input_width = out_width + self.kernel - 1

        dLdA = np.zeros((batch_size, num_channels, input_height, input_width))
        A = self.A

        for i in range(out_height):
            for j in range(out_width):
                window = A[:, :, i:i+self.kernel, j:j+self.kernel]
                window_reshaped = window.reshape(batch_size, num_channels, -1)
                max_indices = np.argmax(window_reshaped, axis=2)
                mask = np.zeros_like(window_reshaped)

                batch_indices = np.arange(batch_size)[:, np.newaxis]
                channel_indices = np.arange(num_channels)[np.newaxis, :]
                mask[batch_indices, channel_indices, max_indices] = 1

                mask = mask.reshape(window.shape)
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += mask * dLdZ[:, :, i, j][:, :, np.newaxis, np.newaxis]

        return dLdA
        


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        out_h = A.shape[2] - self.kernel + 1
        out_w = A.shape[3] - self.kernel + 1
        Z = np.zeros((A.shape[0], A.shape[1], out_h, out_w))
        self.A = A
        
        
        for i in range(out_h):
            for j in range(out_w):
                window = A[:, :, i:i+self.kernel, j:j+self.kernel]
                Z[:, :, i, j] = np.mean(window, axis=(2, 3))
            
        
        return Z
  
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, num_channels, out_height, out_width = dLdZ.shape
        input_height = out_height + self.kernel - 1
        input_width = out_width + self.kernel - 1

        dLdA = np.zeros((batch_size, num_channels, input_height, input_width))

        for i in range(out_height):
                for j in range(out_width):
                    gradient_value = dLdZ[:, :, i, j][:, :, np.newaxis, np.newaxis] / (self.kernel * self.kernel)
                    dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += gradient_value

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        _, _, self.input_width, self.input_height = A.shape
        
        Z = self.maxpool2d_stride1.forward(A)
        Z_downsampled = self.downsample2d.forward(Z)

        return Z_downsampled

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA)
        
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)
   

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        _, _, self.input_width, self.input_height = A.shape
        
        
        Z = self.meanpool2d_stride1.forward(A)
        Z_downsampled = self.downsample2d.forward(Z)

        return Z_downsampled  

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)
        
        return dLdA
