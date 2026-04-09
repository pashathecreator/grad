import numpy as np
from value import Value

class Layer:
    def __init__(self, nin, nout, activation='relu'):
        self.W = Value(np.random.randn(nin, nout) * np.sqrt(2.0 / nin))
        self.b = Value(np.zeros((1, nout)))
        self.activation = activation

    def __call__(self, x):
        if x.data.ndim == 1:
            x = x.reshape(1, -1)
        out = x @ self.W + self.b
        if self.activation == 'relu':
            return out.relu()
        return out

    def parameters(self):
        return [self.W, self.b]
