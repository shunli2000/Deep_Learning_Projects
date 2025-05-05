# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class AdamW():
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.l = model.layers[::2] # every second layer is activation function
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay=weight_decay

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]

    def step(self):

        self.t += 1
        for layer_id, layer in enumerate(self.l):

            # TODO: Calculate updates for weight
            mw = self.m_W[layer_id]
            vw = self.v_W[layer_id]
            
            mw = self.beta1 * mw + (1 - self.beta1) * layer.dLdW
            vw = self.beta2 * vw + (1 - self.beta2) * layer.dLdW * layer.dLdW
            
            # TODO: calculate updates for bias
            mb = self.m_b[layer_id]
            vb = self.v_b[layer_id]
            mb = self.beta1 * mb + (1 - self.beta1) * layer.dLdb
            vb = self.beta2 * vb + (1 - self.beta2) * layer.dLdb * layer.dLdb

            # TODO: Perform weight and bias updates

            mw_ = mw / (1 - self.beta1 ** self.t)
            mb_ = mb / (1 - self.beta1 ** self.t)

            vw_ = vw / (1 - self.beta2 ** self.t)
            vb_ = vb / (1 - self.beta2 ** self.t)

            self.m_W[layer_id] = mw
            self.v_W[layer_id] = vw
            self.m_b[layer_id] = mb
            self.v_b[layer_id] = vb

            layer.W = layer.W - self.lr * mw_ / np.sqrt(vw_ + self.eps) - self.weight_decay * layer.W * self.lr
            layer.b = layer.b - self.lr * mb_ / np.sqrt((vb_ + self.eps)) - self.weight_decay * layer.b * self.lr
            
