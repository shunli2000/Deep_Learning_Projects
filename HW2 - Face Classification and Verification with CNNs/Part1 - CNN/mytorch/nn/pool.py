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
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_size = input_width - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_size, output_size))
        Z_index = np.zeros((batch_size, in_channels, output_size, output_size))

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(output_size):
                    for l in range(output_size):
                        Z[i, j, k, l] = np.max(A[i, j, k:k+self.kernel, l:l+self.kernel])
                        Z_index[i, j, k, l] = np.argmax(A[i, j, k:k+self.kernel, l:l+self.kernel])

        self.Z_index=Z_index
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        _, in_channels, input_width, input_height = self.A.shape

        dLdA=np.zeros_like(self.A)

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(output_width):
                    for l in range(output_height):
                        idx = int(self.Z_index[i, j, k, l])
                        h_idx, w_idx = np.unravel_index(idx, (self.kernel, self.kernel))
                        dLdA[i, j, k+h_idx, l+w_idx] += dLdZ[i, j, k, l]

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
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_size = input_width - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_size, output_size))
        Z_index = np.zeros((batch_size, in_channels, output_size, output_size))

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(output_size):
                    for l in range(output_size):
                        Z[i, j, k, l] = np.average(A[i, j, k:k+self.kernel, l:l+self.kernel])

        return Z       

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape
        _, in_channels, input_width, input_height = self.A.shape

        dLdA=np.zeros_like(self.A)

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(output_width):
                    for l in range(output_height):
                        dLdA[i, j, k:k+self.kernel, l:l+self.kernel] += dLdZ[i, j, k, l] / (self.kernel * self.kernel)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z_unsampled=self.maxpool2d_stride1.forward(A)

        Z=self.downsample2d.forward(Z_unsampled)

        return Z

        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdZ_unsampled=self.downsample2d.backward(dLdZ)

        dLdA=self.maxpool2d_stride1.backward(dLdZ_unsampled)

        return dLdA

        


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1=MeanPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_unsampled=self.meanpool2d_stride1.forward(A)

        Z=self.downsample2d.forward(Z_unsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_unsampled=self.downsample2d.backward(dLdZ)

        dLdA=self.meanpool2d_stride1.backward(dLdZ_unsampled)

        return dLdA
