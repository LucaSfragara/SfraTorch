import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # TODO Create a new array Z with the correct shape
        
        Z = np.zeros((A.shape[0], A.shape[1], 1+ (A.shape[2]-1)*self.upsampling_factor))

        # TODO Fill in the values of Z by upsampling A
        Z[:, :, ::self.upsampling_factor] = A
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        # TODO Slice dLdZ by the upsampling factor to get dLdA

        dLdA = dLdZ[:, :, ::self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # TODO Slice A by the downsampling factor to get Z
        # (hint save any other needed information for backprop later)
        self.W_in = A.shape[2]
        Z = A[:, :, ::self.downsampling_factor]  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        # TODO Create a new array dLdA with the correct shape

        dLdA =  np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.W_in))

        # TODO Fill in the values of dLdA with values of A as needed
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        # TODO Create a new array Z with the correct shape

        Z = np.zeros((A.shape[0], A.shape[1], 1+(A.shape[2]-1)*self.upsampling_factor, 1+(A.shape[3]-1)*self.upsampling_factor))
        # TODO Fill in the values of Z by upsampling A
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # TODO Slice dLdZ by the upsampling factor to get dLdA

        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        # TODO Slice A by the downsampling factor to get Z
        # (hint save any other needed information for backprop later)
        self.H_in = A.shape[2]
        self.W_in = A.shape[3]
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # TODO Create a new array dLdA with the correct shape

        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.H_in, self.W_in))

        # TODO Fill in the values of dLdA with values of A as needed
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ
        
        return dLdA
