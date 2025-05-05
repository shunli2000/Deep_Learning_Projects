import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

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
                # Extract the input patch for each (i, j) position
                input_patch = A[:, :, i:i + self.kernel_size, j:j + self.kernel_size]  # (batch_size, in_channels, kernel_size, kernel_size)
                
                # Compute the dot product with the filters and add bias
                Z[:, :, i, j] = np.tensordot(input_patch, self.W, axes=([1,2, 3], [1, 2, 3])) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, out_channels, output_height, output_width = dLdZ.shape
        _, in_channels, input_height, input_width = self.A.shape

        dLdA = np.zeros(self.A.shape)
        
        self.dLdW = np.zeros_like(self.W)
        self.dLdb = np.zeros_like(self.b)

        # Compute gradients
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(output_height):
                    for l in range(output_width):
                        self.dLdW[j, :, :, :] += dLdZ[i, j, k, l] * self.A[i, :, k:k+self.kernel_size, l:l+self.kernel_size]
                        dLdA[i, :, k:k+self.kernel_size, l:l+self.kernel_size] += dLdZ[i, j, k, l] * self.W[j, :, :, :]

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,weight_init_fn,bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad),(self.pad, self.pad)), mode='constant')

        # Call Conv2d_stride1
        Z_conv = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(Z_conv)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ_down = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA_padded = self.conv2d_stride1.backward(dLdZ_down)

        # Unpad the gradient
        dLdA = dLdA_padded[:, :, self.pad:dLdA_padded.shape[2]-self.pad, self.pad:dLdA_padded.shape[3]-self.pad]

        return dLdA
