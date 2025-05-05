import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        # Save the original shape for use in backward pass
        self.original_shape = A.shape
        
        # Reshape A into (batch_size, in_channels * input_height * input_width)
        Z = A.reshape(A.shape[0], -1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = dLdZ.reshape(self.original_shape)
        
        return dLdA
