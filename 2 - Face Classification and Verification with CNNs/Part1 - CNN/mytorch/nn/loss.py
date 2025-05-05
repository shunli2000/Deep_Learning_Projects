import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO
        se = (A-Y)*(A-Y)  # TODO
        sse = np.ones(self.N).T @ se @ np.ones(self.C) # TODO
        mse = sse/(self.N*self.C)  # TODO

        return mse

    def backward(self):

        dLdA = 2*(self.A-self.Y)/(self.N*self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0] # TODO
        C = A.shape[1]  # TODO

        Ones_C = np.ones(C)  # TODO
        Ones_N = np.ones(N)  # TODO

        self.softmax = (np.exp(A).T/ np.sum(np.exp(A), axis=1).T).T  # TODO
        crossentropy = (-Y*np.log(self.softmax)) @ Ones_C # TODO
        sum_crossentropy = Ones_N @ crossentropy  # TODO
        L = sum_crossentropy / N

        return L

    def backward(self):
        dLdA =( (np.exp(self.A).T/ np.sum(np.exp(self.A), axis=1).T).T - self.Y )/ self.A.shape[0] # TODO

        return dLdA
