# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
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

        batch_size, in_channels, input_length = A.shape
        

        # Output length calculation (stride=1, no padding)
        output_length = input_length - self.kernel_size + 1

        # Initialize output (batch_size, out_channels, output_length)
        Z = np.zeros((batch_size, self.out_channels, output_length))

        for i in range(output_length):
            input_segment = A[:, :, i:i + self.kernel_size]  # (batch_size, in_channels, kernel_size)
            # tensordot axes=([1, 2], [1, 2]) sums over in_channels and kernel_size
            Z[:, :, i] = np.tensordot(input_segment, self.W, axes=([1, 2], [1, 2])) + self.b


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        input_size = output_size + self.kernel_size - 1  # To match input size during forward

        self.dLdb = np.sum(dLdZ, axis=(0, 2))  # Gradient w.r.t bias
        for k in range(self.kernel_size):
            self.dLdW[:, :, k] = np.tensordot(dLdZ, self.A[:, :, k:k+dLdZ.shape[2]], axes=((0,2), (0,2)))

        dLdZ_pad = np.pad(dLdZ, pad_width=((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1)), constant_values=0, mode='constant')
        W_flipped = np.flip(self.W, axis=2)
        dLdA = np.zeros(self.A.shape)
        for i in range(self.A.shape[2]):
            dLdA[:, :, i] = np.tensordot(dLdZ_pad[:, :, i:i+self.kernel_size], W_flipped, axes=((1, 2), (0, 2)))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.padding= padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')

        
        # Call Conv1d_stride1
        Z_conv = self.conv1d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample1d.forward(Z_conv) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ_upsampled = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA_padded = self.conv1d_stride1.backward(dLdZ_upsampled) # TODO

        dLdA = dLdA_padded[:, :,self.padding:dLdA_padded.shape[2]-self.padding]

        return dLdA
